"""One audited RealMLP runner for CPU/CUDA/XLA; no downloads or submissions."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from audit import VERSION
from audit.dataset import Dataset
from audit.io import signature, sha256, write_json, save_npz
from audit.metrics import SCORERS, panel
from audit.model import RealMLP, EMA, flat_anneal
from audit.protocol import jobs
from audit.training import (seed_all, rng_state, restore_rng, fit_scale, apply_scale,
                            loss_parts, gradient_norm, cpu_tree)

DEFAULTS = {'metric':'cosine', 'metric_verified':False, 'metric_source':'',
            'epochs':10, 'patience':3, 'lr':.001, 'batch_size':256, 'n_ens':16,
            'noise_std':.005, 'weight_target':'clean', 'angular_loss':'pearson',
            'lambda_cos':1., 'schedule':'epoch', 'clip_quantiles':[.001,.999],
            'gradient_diagnostics':True, 'keep_completed_last':False}


def read_config(path):
    config = {**DEFAULTS, **json.loads(Path(path).read_text(encoding='utf-8'))}
    if config['metric'] not in SCORERS or config['angular_loss'] not in SCORERS:
        raise ValueError('Unknown score/loss metric')
    if config['metric_verified'] and not config['metric_source'].strip():
        raise ValueError('Record scorer evidence in metric_source')
    if config['mode'] == 'final' and not config['metric_verified']:
        raise ValueError('Verify the official scorer before a final run; screens can report both provisional metrics')
    if config['schedule'] != 'epoch':
        raise ValueError('Audited CPU/CUDA/XLA use the SAME per-epoch schedule')
    if config['weight_target'] not in ('clean','noisy','uniform'):
        raise ValueError('Unknown weight_target')
    if config['epochs'] < 1 or config['patience'] < 1 or config['batch_size'] < 1 or config['n_ens'] < 2:
        raise ValueError('Invalid training dimensions')
    if config['lr'] <= 0 or config['noise_std'] < 0 or config['lambda_cos'] < 0:
        raise ValueError('Invalid optimization configuration')
    q = config['clip_quantiles']
    if q is not None and (len(q) != 2 or not 0 <= q[0] < q[1] <= 1):
        raise ValueError('Invalid clipping quantiles')
    jobs(config)  # Validate protocol before any accelerator work.
    return config


def get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'xla':
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but unavailable')
    return torch.device(name)


def barrier(device):
    if device.type == 'xla':
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def atomic_checkpoint(path, state, device):
    barrier(device)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(cpu_tree(state), tmp)
    os.replace(tmp, path)


def optimizer(model, lr):
    grouped = [[], [], [], []]
    for name, param in model.named_parameters():
        group = 0 if 'scale' in name else 1 if 'num_embed' in name else 3 if 'bias' in name else 2
        grouped[group].append(param)
    return torch.optim.AdamW([
        {'params':grouped[0], 'lr':lr*20, 'weight_decay':.001},
        {'params':grouped[1], 'lr':lr*.093, 'weight_decay':.01},
        {'params':grouped[2], 'lr':lr, 'weight_decay':.01},
        {'params':grouped[3], 'lr':lr*.1, 'weight_decay':.005}], betas=(.9,.98))


def predict(model, x, batch=2048):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x[i:i+batch]).mean(dim=1).cpu().numpy()
                               for i in range(0,len(x),batch)])


def predict_rows(model, data, arm, split, rows, med, fac, device):
    # Feature assembly/inference in bounded blocks; no full test matrix on the device.
    parts = []
    for i in range(0,len(rows),2048):
        x = apply_scale(data.features(arm,split,rows[i:i+2048]),med,fac)
        parts.append(predict(model,torch.from_numpy(x).to(device)))
    return np.concatenate(parts)


def run_job(job, config, data, out, device, run_sig, deadline):
    jobdir = out/'jobs'/job.key
    jobdir.mkdir(parents=True, exist_ok=True)
    done, checkpoint = jobdir/'done.json', jobdir/'last.pt'
    if done.exists():
        meta = json.loads(done.read_text(encoding='utf-8'))
        if meta['run_signature'] != run_sig:
            raise ValueError(f'Incompatible completed job: {job.key}')
        for name,digest in meta['artifacts'].items():
            if sha256(jobdir/name) != digest:
                raise ValueError(f'Corrupted completed artifact: {job.key}/{name}')
        print('REUSE',job.key,flush=True)
        return True
    tr, es, score_mask = job.masks(data.months)
    tr_rows, es_rows, score_rows = [np.flatnonzero(m) for m in (tr,es,score_mask)]
    seed = job.seed*100 + job.fold  # Origin/arm changes do not depend on preceding tasks.
    seed_all(seed,device)
    state = None
    if checkpoint.exists():
        # Only load this runner's own trusted checkpoints, never external pickles.
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['run_signature'] != run_sig or state['job'] != asdict(job):
            raise ValueError('Resume recipe/data/backend mismatch')
    xtr = data.features(job.arm,'train',tr_rows)
    med,fac = (state['med'],state['fac']) if state else fit_scale(xtr)
    xtr = apply_scale(xtr,med,fac)
    xes = apply_scale(data.features(job.arm,'train',es_rows),med,fac)
    ytr = np.asarray(data.y[tr_rows]).copy()
    xtr_t = torch.from_numpy(xtr).to(device)
    xes_t = torch.from_numpy(xes).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    model = RealMLP(xtr.shape[1],config['n_ens']).to(device)
    opt = optimizer(model,config['lr'])
    base_lrs = [g['lr'] for g in opt.param_groups]
    ema = EMA(model,.998)
    history, best, best_state, patience, first_epoch, best_epoch = [], -float('inf'), None, 0, 0, None
    spent = 0.
    if state:
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        ema.ema_state = {k:v.to(device) for k,v in state['ema'].items()}
        best,best_state = state['best'],state['best_state']
        history,patience,first_epoch = state['history'],state['patience'],state['next_epoch']
        best_epoch,spent = state['best_epoch'],state['elapsed_s']
        restore_rng(state['rng'],device)
    del state, xtr, xes
    t0 = time.monotonic()
    status = {'independent_score': config['mode']=='confirm', 'job':asdict(job)}
    for ep in range(first_epoch,config['epochs']):
        if patience >= config['patience']:
            break
        if time.monotonic() >= deadline:
            print('BUDGET_STOP before epoch',job.key,ep,flush=True)
            return False
        model.train()
        progress = ep/max(config['epochs']-1,1)
        for group,base_lr in zip(opt.param_groups,base_lrs):
            group['lr'] = flat_anneal(base_lr,progress)
        # Separate CPU generators keep permutations/noise independent of model dimensions.
        perm = np.random.default_rng(seed*1000+ep).permutation(len(ytr))
        noise_gen = torch.Generator(device='cpu').manual_seed(seed*1000+ep+7000001)
        ep_diag = {}
        for start in range(0,len(perm),config['batch_size']):
            ix_cpu = perm[start:start+config['batch_size']]
            ix = torch.from_numpy(ix_cpu).to(device)
            clean = ytr_t[ix]
            noise = torch.randn(len(ix_cpu),generator=noise_gen).to(device)
            noisy = clean + noise*(config['noise_std']*(1-progress))
            opt.zero_grad(set_to_none=True)
            pred = model(xtr_t[ix])
            loss,mse,angular = loss_parts(pred,clean,noisy,config['weight_target'],
                                         config['angular_loss'],config['lambda_cos'])
            if start == 0 and config['gradient_diagnostics']:
                params = list(model.parameters())
                ep_diag = {'mse':float(mse.detach().cpu()), 'angular':float(angular.detach().cpu()),
                           'mse_grad_norm':float(gradient_norm(mse,params).cpu()),
                           'weighted_angular_grad_norm':float(gradient_norm(config['lambda_cos']*angular,params).cpu())}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            # Independent worker models MUST NOT all-reduce gradients with xm.optimizer_step.
            opt.step()
            ema.update()
            barrier(device)
        original = ema.apply()
        pv = predict(model,xes_t)
        ema.restore(original)
        value = SCORERS[config['metric']](pv,data.y[es_rows])
        if value > best:
            best,best_epoch,patience = value,ep,0
            best_state = cpu_tree(ema.ema_state)
        else:
            patience += 1
        history.append({'epoch':ep, 'es_score':value, 'es_cosine':SCORERS['cosine'](pv,data.y[es_rows]),
                        'es_pearson':SCORERS['pearson'](pv,data.y[es_rows]),
                        'noise_std':config['noise_std']*(1-progress), **ep_diag})
        saved = {'run_signature':run_sig,'job':asdict(job),'model':model.state_dict(),
                 'optimizer':opt.state_dict(),'ema':ema.ema_state,'rng':rng_state(device),
                 'med':med,'fac':fac,'next_epoch':ep+1,'best':best,'best_epoch':best_epoch,
                 'best_state':best_state,'patience':patience,'history':history,
                 'elapsed_s':spent+time.monotonic()-t0}
        atomic_checkpoint(checkpoint,saved,device)
        del saved
        write_json(jobdir/'history.json',history)
        print(job.key,'epoch',ep,'ES',round(value,6),flush=True)
    if best_state is None:
        raise RuntimeError('No checkpoint selected')
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()},strict=False)
    model.eval()
    es_pred = predict(model,xes_t)
    # The independent score is computed ONLY after all checkpoint selection has finished.
    score_pred = (predict_rows(model,data,job.arm,'train',score_rows,med,fac,device)
                  if config['mode']=='confirm' else es_pred.copy())
    arrays = {'es_rows':es_rows,'es_pred':es_pred,'score_rows':score_rows,'score_pred':score_pred}
    if config['mode']=='final':
        test_ids = data.load('test_ids.npy')
        arrays['test_rows'] = np.arange(len(test_ids))
        arrays['test_pred'] = predict_rows(model,data,job.arm,'test',arrays['test_rows'],med,fac,device)
    save_npz(jobdir/'predictions.npz',**arrays)
    atomic_checkpoint(jobdir/'best.pt', {'run_signature':run_sig,'job':asdict(job),
                      'model':model.state_dict(),'med':med,'fac':fac,'best_epoch':best_epoch,
                      'feature_names':data.names(job.arm),'config':config,
                      'dataset_fingerprint':data.manifest['fingerprint']},device)
    meta = {**status,'run_signature':run_sig,'best_epoch':best_epoch,
            'elapsed_s':spent+time.monotonic()-t0,'target_train_std':float(ytr.std()),
            'es_selected_score':best,'n_train':len(tr_rows),'n_es':len(es_rows),
            'artifacts':{name:sha256(jobdir/name) for name in ('predictions.npz','best.pt','history.json')}}
    write_json(done,meta)  # Commit marker last: incomplete jobs can always resume safely.
    if not config['keep_completed_last']:
        # Best model/scaler/predictions are retained; completed optimizer states are large.
        checkpoint.unlink(missing_ok=True)
    del model,opt,ema,xtr_t,xes_t,ytr_t,best_state
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    barrier(device)
    return True


def run(args):
    config = read_config(args.config)
    if args.max_hours <= 0:
        raise ValueError('max-hours must be positive')
    selected = jobs(config)
    if args.workers < 1 or not 0 <= args.worker_id < args.workers:
        raise ValueError('Invalid worker shard')
    selected = selected[args.worker_id::args.workers]
    if args.job_key:
        selected = [j for j in selected if j.key==args.job_key]
        if not selected:
            raise ValueError('Requested job not in shard/config')
    if args.plan:
        print(json.dumps({'models':len(selected),'jobs':[asdict(j) for j in selected],
              'metric_verified':config['metric_verified'],
              'estimate_hours_at_17min_per_model':round(len(selected)*17/60,2)},indent=2))
        return
    start = time.monotonic()
    data = Dataset(args.data)
    for arm in config['arms']:
        data.names(arm)
        data.features(arm,'train',np.array([0]))
        if config['mode']=='final':
            data.features(arm,'test',np.array([0]))
    for job in selected:
        job.masks(data.months)
    device = get_device(args.device)
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    code_files = [Path(__file__),Path(__file__).with_name('summarize_audited.py')] + sorted((Path(__file__).parent/'audit').glob('*.py'))
    provenance = {'audit_version':VERSION,'config':config,'dataset_fingerprint':data.manifest['fingerprint'],
                  'backend':device.type,'torch_version':str(torch.__version__),
                  'code_hashes':{p.name:sha256(p) for p in code_files}}
    if device.type=='xla':
        import torch_xla
        import torch_xla.runtime as xr
        provenance['torch_xla_version'] = str(torch_xla.__version__)
        runtime = {'addressable_devices':xr.addressable_device_count(),'world_size':xr.world_size()}
    else:
        runtime = {'cuda_devices':torch.cuda.device_count()}
    run_sig = signature(provenance)
    manifest_path = out/f'run_worker_{args.worker_id:02d}.json'
    for previous in out.glob('run_worker_*.json'):
        if json.loads(previous.read_text(encoding='utf-8'))['run_signature'] != run_sig:
            raise ValueError('Output directory belongs to another recipe/backend/data version')
    write_json(manifest_path,{**provenance,'run_signature':run_sig,'worker_id':args.worker_id,
                             'workers':args.workers,'device':str(device),'runtime':runtime})
    print('RUN',run_sig,'device',device,'runtime',runtime,'models',len(selected),flush=True)
    if not config['metric_verified']:
        print('PROVISIONAL SCORER: report cosine AND Pearson; final mode requires evidence.',flush=True)
    deadline = start+args.max_hours*3600
    for job in selected:
        if time.monotonic() >= deadline:
            print('BUDGET_STOP before model',job.key,flush=True)
            break
        if not run_job(job,config,data,out,device,run_sig,deadline):
            break
    print('Worker finished. Run summarize_audited.py; partial panels are ineligible.',flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--data',default='prepared')
    p.add_argument('--out',default='runs/audited')
    p.add_argument('--device',choices=['auto','cpu','cuda','xla'],default='auto')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--max-hours',type=float,default=10.)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--worker-id',type=int,default=0)
    p.add_argument('--job-key',help='Isolate a job for order/reproducibility checks')
    return p


if __name__=='__main__':
    run(parser().parse_args())
