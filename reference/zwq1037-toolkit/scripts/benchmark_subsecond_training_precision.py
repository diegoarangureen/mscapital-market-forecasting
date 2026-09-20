"""Measure real event-model batches before choosing training precision."""
from __future__ import annotations
import gc
import json
import os
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from exp_sequence_020_subsecond_gru_transformer import SequenceDataset, make_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER'


def main():
    torch.set_num_threads(2)
    labels = pd.read_feather(ROOT / 'data/raw/label.feather')
    train = np.flatnonzero(labels.month.to_numpy() <= 59)
    rows = np.sort(np.random.default_rng(2026).choice(train, 256, replace=False))
    target = labels.target.to_numpy(np.float32)
    cache = ROOT / 'data/processed/train_subsecond_event_cache_v1'
    static = np.load(ROOT / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')
    market = np.memmap(ROOT / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid/train_v2_market_200x11.mmap', mode='r', dtype=np.float16, shape=(len(labels), 200, 11))
    events = {s: np.load(cache / f'{s}_events.npy', mmap_mode='r') for s in ['transaction', 'order']}
    lengths = {s: np.load(cache / f'{s}_lengths.npy') for s in events}
    stored = np.load(RUN / 'train_fold_normalization.npz')
    norm = {s: (stored[f'{s}_mean'], stored[f'{s}_std']) for s in ['market', 'static', 'transaction', 'order']}
    dataset = SequenceDataset(rows, market, static, events, lengths, norm, target, float(target[train].std()))
    batch = next(iter(DataLoader(dataset, batch_size=256, num_workers=0)))
    inputs, y = [t.to('cuda').contiguous() for t in batch[:4]], batch[4].to('cuda')
    results = []
    for name, dtype in [('float32', None), ('bfloat16', torch.bfloat16), ('float16', torch.float16)]:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        model = make_model().to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
        times, losses = [], []
        for index in range(6):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype or torch.float16, enabled=dtype is not None):
                p = model(*inputs).float()
                pc, yc = p-p.mean(), y-y.mean()
                loss = .35*F.smooth_l1_loss(p, y) + .65*(1-(pc*yc).sum()/(pc.norm()*yc.norm()).clamp_min(1e-8))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if index:
                times.append(time.perf_counter()-started)
                losses.append(float(loss))
        row = {'precision': name, 'seconds_per_batch': float(np.median(times)),
               'losses': losses, 'finite_gradients': True,
               'estimated_four_epoch_train_hours': float(np.median(times))*len(train)//256*4/3600}
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, scaler
        gc.collect(); torch.cuda.empty_cache()
    output = ROOT / 'outputs/submission_metadata/subsecond_precision_benchmark_20260917.json'
    output.write_text(json.dumps({'gpu': torch.cuda.get_device_name(), 'real_batch_size': 256,
                                 'results': results, 'scope': 'Warm repeated real random batch; excludes data loading and validation.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
