"""Strict late-window evaluation of event GRUs inside the old Transformer."""
from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from subsecond_event_warmstart_model import make_warmstart_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/processed/train_subsecond_event_cache_v1'
RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER-WARMSTART'
SEED = 2026
BATCH_SIZE = 256
EPOCHS = 4
OLD_BASELINE = 0.1553845145
SMOKE_BATCHES = int(os.environ.get('SEQ020_SMOKE_BATCHES', '0'))
if SMOKE_BATCHES:
    RUN = RUN.parent / (RUN.name + '-SMOKE')



def make_model():
    model, initialization = make_warmstart_model()
    (RUN / 'initialization.json').write_text(json.dumps(initialization, indent=2), encoding='utf-8')
    return model


def cosine(x, y):
    x = torch.as_tensor(np.asarray(x, dtype=np.float64))
    y = torch.as_tensor(np.asarray(y, dtype=np.float64))
    return float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-20))


def fit_stats(values, lengths, selected):
    total = np.zeros(values.shape[-1], dtype=np.float64)
    squared = total.copy()
    count = 0
    for start in range(0, len(selected), 256):
        rows = selected[start:start + 256]
        block = np.asarray(values[rows], dtype=np.float32)
        valid = np.arange(values.shape[1])[None, :] < lengths[rows, None]
        good = block[valid]
        total += good.sum(axis=0, dtype=np.float64)
        squared += np.square(good).sum(axis=0, dtype=np.float64)
        count += len(good)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(squared / max(count, 1) - mean * mean, 0))
    std[std < 1e-5] = 1
    # 前段标记是离散语义，保留0/1值。
    # Preserve the discrete prefix-summary flag as zero or one.
    mean[-1], std[-1] = 0, 1
    return mean.astype(np.float32), std.astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, rows, market, static, events, lengths, norm, target, scale):
        self.rows, self.market, self.static = rows, market, static
        self.events, self.lengths, self.norm = events, lengths, norm
        self.target, self.scale = target, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = self.rows[position]
        market = np.asarray(self.market[row], dtype=np.float32)
        pad = np.abs(market).sum(axis=-1) == 0
        market = (market - self.norm['market'][0]) / self.norm['market'][1]
        market = np.clip(np.nan_to_num(market), -8, 8)
        market[pad] = 0
        streams = []
        for source in ('transaction', 'order'):
            length = int(self.lengths[source][row])
            # 只读取有效事件，保持原来的固定长度和填充语义。
            # Read valid events only; retain identical fixed shapes and padding.
            raw = np.asarray(self.events[source][row, :length], dtype=np.float32)
            stream = np.zeros((self.events[source].shape[1], raw.shape[1] + 1), dtype=np.float32)
            stream[:length, :-1] = np.clip((raw - self.norm[source][0]) / self.norm[source][1], -8, 8)
            stream[:length, -1] = 1
            streams.append(torch.from_numpy(stream))
        static = np.asarray(self.static[row], dtype=np.float32)
        static = (static - self.norm['static'][0]) / self.norm['static'][1]
        static = np.clip(np.nan_to_num(static), -8, 8).astype(np.float32)
        return torch.from_numpy(market.astype(np.float32)), *streams, torch.from_numpy(static), torch.tensor(np.float32(self.target[row] / self.scale))



def prefetched_batches(loader):
    # 只预读一批，线程共享mmap，避免Windows进程复制大缓存。
    # Prefetch one batch in a shared-memory thread, never pickle large memmaps.
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(next, iterator, None)
        while True:
            batch = pending.result()
            if batch is None:
                break
            pending = executor.submit(next, iterator, None)
            yield batch

@torch.no_grad()
def predict(model, loader, device, scale):
    model.eval()
    result = []
    for batch in loader:
        inputs = [item.to(device, non_blocking=True).contiguous() for item in batch[:4]]
        prediction = model(*inputs)
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite validation predictions')
        result.append(prediction.cpu().numpy())
    return np.concatenate(result) * scale


def main():
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    RUN.mkdir(parents=True, exist_ok=True)
    labels = pd.read_feather(ROOT / 'data/raw/label.feather')
    months, target = labels['month'].to_numpy(), labels['target'].to_numpy(dtype=np.float32)
    ids = labels['sample_id'].to_numpy()
    assert np.array_equal(ids, np.load(CACHE / 'sample_ids.npy'))
    train_rows = np.flatnonzero(months <= 59)
    valid_rows = np.flatnonzero((months >= 62) & (months <= 70) & (months != 66))
    if SMOKE_BATCHES:
        valid_rows = valid_rows[:BATCH_SIZE]
    grid = ROOT / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid/train_v2_market_200x11.mmap'
    market = np.memmap(grid, mode='r', dtype=np.float16, shape=(len(ids), 200, 11))
    static = np.load(ROOT / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')
    assert static.shape == (len(ids), 379)
    events = {name: np.load(CACHE / f'{name}_events.npy', mmap_mode='r') for name in ('transaction', 'order')}
    lengths = {name: np.load(CACHE / f'{name}_lengths.npy') for name in events}
    reference = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v7_baseline.py'
    tree = ast.parse(reference.read_text(encoding='utf-8'))
    bundled = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BUNDLED_STREAM_NORM' for t in node.targets))
    rng = np.random.default_rng(SEED + 101)
    selected = np.sort(rng.choice(train_rows, min(100_000, len(train_rows)), replace=False))
    static_mean = np.zeros(379, dtype=np.float64)
    static_square = static_mean.copy()
    static_count = static_mean.copy()
    for start in range(0, len(selected), 4096):
        block = np.asarray(static[selected[start:start + 4096]], dtype=np.float32)
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0)
        static_mean += safe.sum(axis=0, dtype=np.float64)
        static_square += np.square(safe).sum(axis=0, dtype=np.float64)
        static_count += finite.sum(axis=0)
    static_mean /= np.maximum(static_count, 1)
    static_std = np.sqrt(np.maximum(static_square / np.maximum(static_count, 1) - static_mean ** 2, 0))
    static_std[static_std < 1e-5] = 1
    norm = {
        'market': (np.asarray(bundled['market']['mean'], dtype=np.float32), np.asarray(bundled['market']['std'], dtype=np.float32)),
        'static': (static_mean.astype(np.float32), static_std.astype(np.float32)),
    }
    event_selected = np.sort(np.random.default_rng(SEED + 102).choice(train_rows, min(10_000, len(train_rows)), replace=False))
    for source in events:
        norm[source] = fit_stats(events[source], lengths[source], event_selected)
    np.savez(RUN / 'train_fold_normalization.npz', **{f'{name}_{stat}': values[i] for name, values in norm.items() for i, stat in enumerate(('mean', 'std'))})
    scale = float(target[train_rows].std())
    train_loader = DataLoader(SequenceDataset(train_rows, market, static, events, lengths, norm, target, scale), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(SequenceDataset(valid_rows, market, static, events, lengths, norm, target, scale), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    device = torch.device('cuda')
    model = make_model().to(device)
    model.eval()
    smoke = next(iter(train_loader))
    with torch.no_grad():
        prediction = model(*[x.to(device).contiguous() for x in smoke[:4]])
    assert torch.isfinite(prediction).all()
    print('real_batch_eval_smoke=passed', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best, logs = -1.0, []
    recovery_path = RUN / 'recovery_checkpoint.pt'
    config = {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
              'smoke_batches': SMOKE_BATCHES, 'training_precision': 'float32', 'train_end': 59, 'valid_start': 62,
              'excluded_month': 66, 'cache': str(CACHE), 'initialization': 'strict_old_transformer_market_static_fusion_head', 'teacher_sha256': '85add86334f0912f977a4240a6a0826a50658219a11b0943431f036b68527682'}
    recovery = None
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location='cpu', weights_only=False)
        previous_config = dict(recovery['config'])
        previous_precision = previous_config.pop('training_precision', 'float32')
        expected_config = dict(config)
        expected_config.pop('training_precision')
        if previous_config != expected_config or previous_precision not in ('bfloat16', 'float32'):
            raise RuntimeError('Checkpoint configuration mismatch')
        if previous_precision != 'float32':
            print('precision migration: saved BF16 training -> float32; model/optimizer retained', flush=True)
            (RUN / 'precision_migration.json').write_text(json.dumps({
                'from': previous_precision, 'to': 'float32',
                'epoch': recovery['epoch'], 'next_batch': recovery['next_batch'],
                'reason': 'GPU driver recovered after cuDNN internal error',
                'model_and_optimizer_preserved': True,
            }, indent=2), encoding='utf-8')
        model.load_state_dict(recovery['model'])
        optimizer.load_state_dict(recovery['optimizer'])
        scheduler.load_state_dict(recovery['scheduler'])
        best, logs = recovery['best'], recovery['logs']
        torch.set_rng_state(recovery['torch_rng'])
        torch.cuda.set_rng_state_all(recovery['cuda_rng'])
        print(f"resume epoch={recovery['epoch']} batch={recovery['next_batch']} stage={recovery['stage']}", flush=True)

    def save_recovery(epoch, next_batch, stage, total, count, seconds):
        temporary = recovery_path.with_suffix('.tmp')
        torch.save({'config': config, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'epoch': epoch, 'next_batch': next_batch, 'stage': stage,
                    'total_loss': total, 'count': count, 'seconds': seconds,
                    'best': best, 'logs': logs, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
        temporary.replace(recovery_path)

    start_epoch = recovery['epoch'] if recovery else 1
    compact = {'experiment': 'EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER',
               'best_cosine': best, 'baseline': OLD_BASELINE, 'delta': best-OLD_BASELINE,
               'epochs_finished': len(logs), 'logs': logs, 'passed': False,
               'smoke_only': bool(SMOKE_BATCHES)}
    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.time()
        state = recovery if recovery and epoch == start_epoch else None
        skip = int(state['next_batch']) if state else 0
        total = float(state['total_loss']) if state else 0.0
        count = int(state['count']) if state else 0
        prior_seconds = float(state['seconds']) if state else 0.0
        validation_only = state and state['stage'] == 'validation'
        if not validation_only:
            shuffled = np.random.default_rng(SEED + epoch).permutation(train_rows)
            remaining_rows = shuffled[skip * BATCH_SIZE:]
            loader_generator = torch.Generator().manual_seed(SEED + epoch)
            epoch_loader = DataLoader(SequenceDataset(remaining_rows, market, static, events, lengths, norm, target, scale),
                                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                      pin_memory=False, drop_last=True, generator=loader_generator)
            model.train()
            if not state:
                save_recovery(epoch, 0, 'train', 0, 0, 0)
            for batch_index, batch in enumerate(prefetched_batches(epoch_loader), start=skip):
                inputs = [x.to(device, non_blocking=True).contiguous() for x in batch[:4]]
                y = batch[4].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', enabled=False):
                    p = model(*inputs)
                p = p.float()
                pc, yc = p - p.mean(), y - y.mean()
                cosine_loss = 1 - (pc * yc).sum() / (pc.norm() * yc.norm()).clamp_min(1e-8)
                loss = .35 * F.smooth_l1_loss(p, y) + .65 * cosine_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total += float(loss.detach()) * len(y)
                count += len(y)
                if (batch_index + 1) % 250 == 0 or (SMOKE_BATCHES and batch_index + 1 >= SMOKE_BATCHES):
                    save_recovery(epoch, batch_index + 1, 'train', total, count, prior_seconds + time.time()-started)
                    print(f'checkpoint epoch={epoch} batches={batch_index+1} seconds={prior_seconds+time.time()-started:.1f}', flush=True)
                if SMOKE_BATCHES and batch_index + 1 >= SMOKE_BATCHES:
                    break
            scheduler.step()
            save_recovery(epoch, 0, 'validation', total, count, prior_seconds + time.time()-started)
        validation = predict(model, valid_loader, device, scale)
        score = cosine(validation, target[valid_rows])
        log = {'epoch': epoch, 'loss': total / max(count, 1), 'cosine': score,
               'seconds': prior_seconds + time.time() - started}
        logs.append(log)
        print(json.dumps(log), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), RUN / 'best_model.pt')
            pd.DataFrame({'sample_id': ids[valid_rows], 'month': months[valid_rows],
                          'target': target[valid_rows], 'prediction': validation}).to_feather(RUN / 'validation_predictions.feather')
        compact = {'experiment': 'EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER',
                   'best_cosine': best, 'baseline': OLD_BASELINE, 'delta': best-OLD_BASELINE,
                   'epochs_finished': epoch, 'logs': logs,
                   'passed': bool(not SMOKE_BATCHES and best >= OLD_BASELINE+.001),
                   'smoke_only': bool(SMOKE_BATCHES), 'checkpoint': str(recovery_path)}
        (RUN / 'score_only.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')
        save_recovery(epoch + 1, 0, 'train', 0, 0, 0)
        recovery = None
        if SMOKE_BATCHES or (epoch == 1 and log['seconds'] * EPOCHS > 7200):
            compact['termination'] = 'smoke_completed' if SMOKE_BATCHES else 'runtime_budget_exceeded_no_expansion'
            print(compact['termination'], flush=True)
            break
    (RUN / 'result.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
