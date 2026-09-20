"""Train the evidence-backed event model on all labels and build a fixed 75/25 CSV."""
from __future__ import annotations

import ast
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from exp_sequence_020_subsecond_gru_transformer_fp32 import SequenceDataset, fit_stats
from subsecond_event_transformer_model import make_model

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CACHE = ROOT / 'data/processed/train_subsecond_event_cache_v1'
TEST_CACHE = ROOT / 'data/processed/test_subsecond_event_cache_v1'
GRID = ROOT / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid/train_v2_market_200x11.mmap'
TEST_GRID = ROOT / 'data/processed/subsecond_fulltrain_grid/test_v2_market_200x11.mmap'
TRAIN_STATIC = ROOT / 'data/interim/our379_reference_cache/features.npy'
RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-022-SUBSECOND-FULL-E3'
SEED = 2026
BATCH_SIZE = 256
EPOCHS = 3
OLD_DEV_RMS = 0.00033945538892300123
NEW_DEV_RMS = 0.000395204559828322


def prefetched_batches(loader):
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(next, iterator, None)
        while True:
            batch = pending.result()
            if batch is None:
                break
            pending = executor.submit(next, iterator, None)
            yield batch


def fit_static_stats(values, rows):
    total = np.zeros(379, dtype=np.float64)
    squared = total.copy()
    count = total.copy()
    for start in range(0, len(rows), 4096):
        block = np.asarray(values[rows[start:start + 4096]], dtype=np.float32)
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0)
        total += safe.sum(axis=0, dtype=np.float64)
        squared += np.square(safe).sum(axis=0, dtype=np.float64)
        count += finite.sum(axis=0)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(squared / np.maximum(count, 1) - mean * mean, 0))
    std[std < 1e-5] = 1
    return mean.astype(np.float32), std.astype(np.float32)


class CombinedTestStatic:
    def __init__(self, expected_ids):
        base_dir = ROOT / 'data/interim/kaggle_relative319_test'
        self.base = np.load(base_dir / 'test_features.npy', mmap_mode='r')
        ids = np.load(base_dir / 'test_sample_ids.npy', mmap_mode='r')
        if self.base.shape != (len(expected_ids), 319) or not np.array_equal(ids, expected_ids):
            raise AssertionError('Relative319 test input does not align')
        columns = json.loads((ROOT / 'data/interim/kaggle_relative319_dev/feature_columns.json').read_text(encoding='utf-8'))
        top = [
            'trade_volume_imbalance_20', 'last60_ask_price_1_end', 'ofi_1_20_mean',
            'trade_count_imbalance_60', 'new_order_volume_imbalance_10',
            'net_order_pressure_30', 'ofi_1_last', 'microprice_displacement_last',
            'bid_price_1_last', 'ask_price_1_last', 'trade_volume_imbalance_60',
            'net_order_pressure_10', 'last_trade_seconds_before_predict',
            'microprice_displacement_20_mean', 'spread_1_mean',
            'cancel_order_pressure_10', 'last60_spread_2_mean',
            'last60_bid_price_2_end', 'bid_price_2_last', 'bid_price_2_std']
        indices = [columns.index(name) for name in top]
        self.xs40 = np.zeros((len(expected_ids), 40), dtype=np.float32)
        for column, source_index in enumerate(indices):
            vector = np.asarray(self.base[:, source_index], dtype=np.float32)
            finite = np.isfinite(vector)
            values = vector[finite]
            ranks = rankdata(values, method='average').astype(np.float32)
            self.xs40[finite, column] = 2 * ranks / (len(values) + 1) - 1
            std = max(float(values.std(dtype=np.float64)), 1e-8)
            self.xs40[finite, 20 + column] = np.clip((values - float(values.mean(dtype=np.float64))) / std, -10, 10)
        table = pd.read_feather(ROOT / 'data/processed/test_order_quote_position_features.feather')
        if not np.array_equal(table['sample_id'].to_numpy(), expected_ids):
            raise AssertionError('Order20 test input does not align')
        self.order = table.drop(columns=['sample_id']).to_numpy(dtype=np.float32)
        if self.order.shape != (len(expected_ids), 20):
            raise AssertionError('Unexpected Order20 shape')
        self.shape = (len(expected_ids), 379)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return np.concatenate((np.asarray(self.base[index], dtype=np.float32),
                               np.asarray(self.xs40[index], dtype=np.float32),
                               np.asarray(self.order[index], dtype=np.float32)))


class TestDataset(Dataset):
    def __init__(self, market, static, events, lengths, norm):
        self.market, self.static, self.events, self.lengths, self.norm = market, static, events, lengths, norm

    def __len__(self):
        return len(self.static)

    def __getitem__(self, row):
        market = np.asarray(self.market[row], dtype=np.float32)
        padding = np.abs(market).sum(axis=-1) == 0
        market = np.clip(np.nan_to_num((market-self.norm['market'][0])/self.norm['market'][1]), -8, 8)
        market[padding] = 0
        streams = []
        for source in ('transaction', 'order'):
            length = int(self.lengths[source][row])
            raw = np.asarray(self.events[source][row, :length], dtype=np.float32)
            stream = np.zeros((self.events[source].shape[1], raw.shape[1]+1), dtype=np.float32)
            stream[:length, :-1] = np.clip((raw-self.norm[source][0])/self.norm[source][1], -8, 8)
            stream[:length, -1] = 1
            streams.append(torch.from_numpy(stream))
        static = np.asarray(self.static[row], dtype=np.float32)
        static = np.clip(np.nan_to_num((static-self.norm['static'][0])/self.norm['static'][1]), -8, 8).astype(np.float32)
        return torch.from_numpy(market.astype(np.float32)), *streams, torch.from_numpy(static)


@torch.no_grad()
def predict(model, loader, device, scale):
    model.eval()
    outputs = []
    for batch in prefetched_batches(loader):
        prediction = model(*[value.to(device).contiguous() for value in batch])
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite test prediction')
        outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs).astype(np.float64) * scale


def main():
    torch.set_num_threads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    for required in (TRAIN_CACHE / 'order_events.npy', TEST_CACHE / 'order_events.npy', TEST_GRID, TRAIN_STATIC):
        if not required.exists():
            raise FileNotFoundError(required)
    RUN.mkdir(parents=True, exist_ok=True)
    labels = pd.read_feather(ROOT / 'data/raw/label.feather')
    train_ids = labels.sample_id.to_numpy(dtype=np.int64)
    target = labels.target.to_numpy(dtype=np.float32)
    rows = np.arange(len(labels), dtype=np.int64)
    if not np.array_equal(train_ids, np.load(TRAIN_CACHE / 'sample_ids.npy')):
        raise AssertionError('Train event IDs do not align')
    market = np.memmap(GRID, mode='r', dtype=np.float16, shape=(len(labels), 200, 11))
    static = np.load(TRAIN_STATIC, mmap_mode='r')
    events = {name: np.load(TRAIN_CACHE / f'{name}_events.npy', mmap_mode='r') for name in ('transaction','order')}
    lengths = {name: np.load(TRAIN_CACHE / f'{name}_lengths.npy') for name in events}
    reference = ast.parse((ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v7_baseline.py').read_text(encoding='utf-8'))
    bundled = next(ast.literal_eval(node.value) for node in reference.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BUNDLED_STREAM_NORM' for t in node.targets))
    selected = np.sort(np.random.default_rng(SEED+101).choice(rows, 100_000, replace=False))
    norm = {'market': (np.asarray(bundled['market']['mean'], dtype=np.float32), np.asarray(bundled['market']['std'], dtype=np.float32)),
            'static': fit_static_stats(static, selected)}
    event_selected = np.sort(np.random.default_rng(SEED+102).choice(rows, 10_000, replace=False))
    for source in events:
        norm[source] = fit_stats(events[source], lengths[source], event_selected)
    np.savez(RUN / 'fulltrain_normalization.npz', **{f'{name}_{stat}': values[i] for name, values in norm.items() for i,stat in enumerate(('mean','std'))})
    scale = float(target.std())
    device = torch.device('cuda')
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    recovery_path = RUN / 'recovery_checkpoint.pt'
    config = {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'precision': 'float32',
              'train_rows': len(rows), 'fixed_new_weight': 0.25}
    recovery = torch.load(recovery_path, map_location='cpu', weights_only=False) if recovery_path.exists() else None
    if recovery:
        if recovery['config'] != config:
            raise RuntimeError('Fulltrain checkpoint config mismatch')
        model.load_state_dict(recovery['model']); optimizer.load_state_dict(recovery['optimizer']); scheduler.load_state_dict(recovery['scheduler'])
        torch.set_rng_state(recovery['torch_rng']); torch.cuda.set_rng_state_all(recovery['cuda_rng'])
        print(f"resume epoch={recovery['epoch']} batch={recovery['next_batch']} stage={recovery['stage']}", flush=True)

    def save(epoch, next_batch, stage, total, count, seconds):
        temporary = recovery_path.with_suffix('.tmp')
        torch.save({'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'next_batch': next_batch,
                    'stage': stage, 'total': total, 'count': count, 'seconds': seconds,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
        temporary.replace(recovery_path)

    start_epoch = recovery['epoch'] if recovery else 1
    training_log = []
    log_path = RUN / 'training_log.json'
    if log_path.exists(): training_log = json.loads(log_path.read_text())
    for epoch in range(start_epoch, EPOCHS+1):
        state = recovery if recovery and epoch == start_epoch else None
        if state and state['stage'] == 'inference':
            break
        started = time.time(); skip = int(state['next_batch']) if state else 0
        total = float(state['total']) if state else 0.0; count = int(state['count']) if state else 0
        prior = float(state['seconds']) if state else 0.0
        shuffled = np.random.default_rng(SEED+epoch).permutation(rows)
        remaining = shuffled[skip*BATCH_SIZE:]
        dataset = SequenceDataset(remaining, market, static, events, lengths, norm, target, scale)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
        model.train()
        if not state: save(epoch, 0, 'train', 0, 0, 0)
        for batch_index, batch in enumerate(prefetched_batches(loader), start=skip):
            inputs = [value.to(device).contiguous() for value in batch[:4]]; y = batch[4].to(device)
            optimizer.zero_grad(set_to_none=True); prediction = model(*inputs).float()
            centered_prediction = prediction-prediction.mean(); centered_target = y-y.mean()
            cosine_loss = 1-(centered_prediction*centered_target).sum()/(centered_prediction.norm()*centered_target.norm()).clamp_min(1e-8)
            loss = .35*F.smooth_l1_loss(prediction,y)+.65*cosine_loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite fulltrain loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            total += float(loss.detach())*len(y); count += len(y)
            if (batch_index+1)%250 == 0:
                save(epoch,batch_index+1,'train',total,count,prior+time.time()-started)
                print(f'checkpoint epoch={epoch} batches={batch_index+1} seconds={prior+time.time()-started:.1f}',flush=True)
        scheduler.step()
        entry = {'epoch': epoch, 'loss': total/max(count,1), 'seconds': prior+time.time()-started}
        training_log.append(entry); log_path.write_text(json.dumps(training_log,indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True); save(epoch+1,0,'inference' if epoch==EPOCHS else 'train',0,0,0)
        recovery = None
    torch.save({'model':model.state_dict(),'target_scale':scale,'epochs':EPOCHS,'seed':SEED}, RUN/'subsecond_full_e3.pt')

    test_ids = pd.read_csv(ROOT/'data/raw/submission.csv',usecols=['sample_id']).sample_id.to_numpy(dtype=np.int64)
    if not np.array_equal(test_ids,np.load(TEST_CACHE/'sample_ids.npy')): raise AssertionError('Test event IDs do not align')
    test_market = np.memmap(TEST_GRID,mode='r',dtype=np.float16,shape=(len(test_ids),200,11))
    test_static = CombinedTestStatic(test_ids)
    test_events = {name:np.load(TEST_CACHE/f'{name}_events.npy',mmap_mode='r') for name in ('transaction','order')}
    test_lengths = {name:np.load(TEST_CACHE/f'{name}_lengths.npy') for name in test_events}
    test_loader = DataLoader(TestDataset(test_market,test_static,test_events,test_lengths,norm),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=False)
    new_prediction = predict(model,test_loader,device,scale)
    raw_path = ROOT/'outputs/submissions/standalone_subsecond_event_transformer379_full_e3.csv'
    pd.DataFrame({'sample_id':test_ids,'prediction':new_prediction}).to_csv(raw_path,index=False)
    old_path = ROOT/'data/interim/kaggle_outputs/multistream_factorized_transformer_fulltrain/factorized_transformer/factorized_transformer379_full_e4.csv'
    old = pd.read_csv(old_path)
    if not np.array_equal(old.sample_id.to_numpy(),test_ids): raise AssertionError('Old full prediction IDs do not align')
    mixed = .75*old.prediction.to_numpy(dtype=np.float64)/OLD_DEV_RMS + .25*new_prediction/NEW_DEV_RMS
    old_full_std = float(old.prediction.std(ddof=0)); mixed *= old_full_std/float(mixed.std(ddof=0))
    blend_path = ROOT/'outputs/submissions/fixed75_transformer25_subsecond_event_full.csv'
    pd.DataFrame({'sample_id':test_ids,'prediction':mixed}).to_csv(blend_path,index=False)
    report = {'experiment':'EXP-SEQUENCE-022-SUBSECOND-FULL-E3','train_rows':len(rows),'epochs':EPOCHS,
              'new_prediction_file':str(raw_path),'blend_prediction_file':str(blend_path),
              'fixed_weights':{'old_transformer':.75,'subsecond_event':.25},'weight_search':False,
              'dev_evidence':{'selection_delta':.004723717199394628,'forward_delta':.003117017650522441,'all_delta':.003623365670093026},
              'normalization':{'old_dev_rms':OLD_DEV_RMS,'new_dev_rms':NEW_DEV_RMS,'final_std':old_full_std},
              'new_mean':float(new_prediction.mean()),'new_std':float(new_prediction.std()),
              'blend_mean':float(mixed.mean()),'blend_std':float(mixed.std()),'rows':len(test_ids),
              'formal_submission':False}
    (ROOT/'outputs/submission_metadata/fixed75_transformer25_subsecond_full_20260917.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
