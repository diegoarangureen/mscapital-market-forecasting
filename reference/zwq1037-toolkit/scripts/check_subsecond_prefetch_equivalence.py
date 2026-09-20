"""Check real cached samples and checkpoint compatibility before resuming."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import exp_sequence_020_subsecond_gru_transformer_bfloat16 as old
import exp_sequence_020_subsecond_gru_transformer_prefetch as new

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
labels = pd.read_feather(root / 'data/raw/label.feather')
target = labels.target.to_numpy(np.float32)
rows = np.sort(np.random.default_rng(2026).choice(np.flatnonzero(labels.month.to_numpy() <= 59), 24, replace=False))
cache = root / 'data/processed/train_subsecond_event_cache_v1'
events = {s: np.load(cache / f'{s}_events.npy', mmap_mode='r') for s in ['transaction', 'order']}
lengths = {s: np.load(cache / f'{s}_lengths.npy') for s in events}
static = np.load(root / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')
market = np.memmap(root / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid/train_v2_market_200x11.mmap', mode='r', dtype=np.float16, shape=(len(labels), 200, 11))
saved = np.load(old.RUN / 'train_fold_normalization.npz')
norm = {s: (saved[f'{s}_mean'], saved[f'{s}_std']) for s in ['market', 'static', 'transaction', 'order']}
args = (rows, market, static, events, lengths, norm, target, .0025)
a, b = old.SequenceDataset(*args), new.SequenceDataset(*args)
for index in range(len(rows)):
    assert all(torch.equal(x, y) for x, y in zip(a[index], b[index]))
ordinary = DataLoader(a, batch_size=8, shuffle=False, num_workers=0)
prefetched = new.prefetched_batches(DataLoader(b, batch_size=8, shuffle=False, num_workers=0))
for x, y in zip(ordinary, prefetched):
    assert all(torch.equal(u, v) for u, v in zip(x, y))
state = torch.load(old.RUN / 'recovery_checkpoint.pt', map_location='cpu', weights_only=False)
assert state['config']['training_precision'] == 'bfloat16'
assert state['config']['batch_size'] == 256
print({'cached_samples_identical': len(rows), 'prefetched_batch_order_identical': True,
       'checkpoint_epoch': state['epoch'], 'checkpoint_next_batch': state['next_batch'],
       'checkpoint_stage': state['stage']}, flush=True)
