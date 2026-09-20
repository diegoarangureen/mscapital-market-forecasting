"""Recover temporal TabM/GRU validation predictions from existing checkpoints."""
from __future__ import annotations
import ast
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from search_owned_five_model_grid import load_existing

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/interim/tree_experiments/EXP-BLEND-OWNED-20260917/recovered_temporal'
FOLDS = (52, 57, 62)
SEEDS = (42, 137, 2026)
MARKET6 = 'mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 realized_volatility_60 mid_momentum_60'.split()


def write_prediction(path, ids, prediction):
    assert prediction.shape == (len(ids),) and np.isfinite(prediction).all()
    pd.DataFrame({'sample_id': ids, 'prediction': prediction}).to_feather(path)


def infer_tabm(static, rows, ids):
    sys.path.insert(0, str(ROOT / 'data/interim/kaggle_kernels/relative319_xs_tabm_notebook'))
    import run_extracted as recipe
    features = np.empty((len(rows), 385), np.float32)
    features[:, :379] = static[rows]
    market = pd.read_feather(ROOT / 'data/processed/train_market_microstructure_features.feather', columns=['sample_id', *MARKET6]).sort_values('sample_id')
    assert np.array_equal(market.sample_id.to_numpy()[rows], ids)
    features[:, 379:] = market[MARKET6].to_numpy(np.float32)[rows]
    del market
    gc.collect()
    names = json.loads((ROOT / 'data/interim/our379_reference_cache/feature_columns.json').read_text(encoding='utf-8')) + MARKET6
    predictions = {}
    model_root = ROOT / 'data/interim/submissions/tabm385_k32_temporal3fold_3seed'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for fold in FOLDS:
        fold_dir = model_root / f'train000_{fold:03d}'
        prep = np.load(fold_dir / 'quantile_preprocessing.npz')
        assert prep['feature_columns'].tolist() == names
        preprocessor = recipe.QuantilePreprocessor(prep['knots'], prep['medians'], prep['missing_columns'], torch.device('cuda'))
        for seed in SEEDS:
            path = RUN / f'tabm_fold{fold}_seed{seed}.feather'
            if path.exists():
                saved = pd.read_feather(path)
                assert np.array_equal(saved.sample_id.to_numpy(), ids)
                predictions[fold, seed] = saved.prediction.to_numpy()
                continue
            started = time.time()
            model = recipe.TabM.make(n_num_features=preprocessor.output_dimension, cat_cardinalities=None, d_out=1, k=32, n_blocks=2, d_block=256, dropout=.1, arch_type='tabm').to('cuda')
            model_dir = fold_dir / f'seed_{seed}'
            model.load_state_dict(torch.load(model_dir / 'model.pt', map_location='cuda', weights_only=True))
            stats = json.loads((model_dir / 'result.json').read_text(encoding='utf-8'))
            result = recipe.predict(model, features, np.arange(len(features)), preprocessor, amp_dtype).astype(np.float64) * stats['target_std']
            write_prediction(path, ids, result)
            predictions[fold, seed] = result
            print(f'tabm fold={fold} seed={seed} recovered seconds={time.time()-started:.1f}', flush=True)
            del model
            torch.cuda.empty_cache()
        del preprocessor
    return predictions


def infer_gru(static, rows, ids, months):
    source_path = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v12_gru_temporal3fold.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    definitions = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    norm_literal = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BUNDLED_STREAM_NORM' for t in node.targets))
    norm = {name: {k: np.asarray(v, np.float32) for k, v in stats.items()} for name, stats in norm_literal.items()}
    namespace = {'np': np, 'torch': torch, 'nn': nn, 'math': math, 'Dataset': Dataset, 'SEED': 2026,
                 'MARKET_FEATURES': range(11), 'TX_FEATURES': range(7), 'ORDER_FEATURES': range(10),
                 'MARKET_LEN': 200, 'FLOW_LEN': 60, 'MARKET_SECONDS': 600., 'FLOW_SECONDS': 60.,
                 'STATIC_FEATURE_COUNT': 379, '_STATIC_FEATURES': static, '_STATIC_NORM': None}
    for name in ['_compute_static_norm', 'GridDataset', '_TimeAwareGRUStreamEncoder', '_JointMultiStreamStaticModel']:
        exec(definitions[name], namespace)
    grid = ROOT / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid'
    arrays = {name: np.memmap(grid / file, mode='r', dtype=np.float16, shape=(len(months), length, width)) for name, file, length, width in [
        ('market', 'train_v2_market_200x11.mmap', 200, 11),
        ('tx', 'train_v2_tx_60x7.mmap', 60, 7),
        ('order', 'train_v2_order_60x10.mmap', 60, 10)]}
    predictions = {}
    model_root = ROOT / 'data/interim/kaggle_outputs/multistream_factorized_gru_temporal3fold_v12/factorized_gru_fulltrain'
    for fold in FOLDS:
        path = RUN / f'gru_fold{fold}.feather'
        if path.exists():
            saved = pd.read_feather(path)
            assert np.array_equal(saved.sample_id.to_numpy(), ids)
            predictions[fold] = saved.prediction.to_numpy()
            continue
        started = time.time()
        namespace['_STATIC_NORM'] = namespace['_compute_static_norm'](static, np.flatnonzero(months <= fold))
        dataset = namespace['GridDataset'](arrays, rows, norm)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
        state = torch.load(model_root / f'factorized_gru_fold_end{fold}_e4.pt', map_location='cpu', weights_only=True)
        model = namespace['_JointMultiStreamStaticModel']().to('cuda')
        weights = {k.removeprefix('parallel.module.').removeprefix('parallel.'): v for k, v in state['model'].items()}
        model.load_state_dict(weights)
        model.eval()
        parts = []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                prediction = model(*[x.to('cuda', non_blocking=True).contiguous() for x in batch])
                assert torch.isfinite(prediction).all()
                parts.append(prediction.cpu().numpy())
                if (index + 1) % 100 == 0:
                    print(f'gru fold={fold} batches={index+1}/{len(loader)} seconds={time.time()-started:.1f}', flush=True)
        result = np.concatenate(parts).astype(np.float64) * state['target_scale']
        write_prediction(path, ids, result)
        predictions[fold] = result
        print(f'gru fold={fold} recovered seconds={time.time()-started:.1f}', flush=True)
        del model, state, loader
        torch.cuda.empty_cache()
        gc.collect()
    return predictions


def main():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    assert torch.cuda.is_available()
    RUN.mkdir(parents=True, exist_ok=True)
    labels = pd.read_feather(ROOT / 'data/raw/label.feather').sort_values('sample_id').reset_index(drop=True)
    cache_ids = np.load(ROOT / 'data/interim/kaggle_relative319_dev/sample_ids.npy', mmap_mode='r')
    assert np.array_equal(cache_ids, labels.sample_id.to_numpy())
    months = labels.month.to_numpy()
    rows = np.flatnonzero((months >= 62) & (months <= 70) & (months != 66))
    ids = labels.sample_id.to_numpy()[rows]
    static = np.load(ROOT / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')
    tabm = infer_tabm(static, rows, ids)
    gru = infer_gru(static, rows, ids, months)
    old = load_existing()
    assert np.array_equal(old.sample_id.to_numpy(), ids)
    # 截止57月的两折可用于62月起验证；含62月训练的三折只用于65月起。
    # Enforce the two-month gap for all contributing checkpoints.
    two = old.copy()
    two['tabm'] = np.mean([p for (fold, seed), p in tabm.items() if fold <= 57], axis=0)
    def zunit(a, fit_mask):
        return (a - a[fit_mask].mean()) / max(float(a[fit_mask].std()), 1e-12)
    two['gru'] = np.mean([zunit(p, old.month.to_numpy() <= 65) for fold, p in gru.items() if fold <= 57], axis=0)
    two.to_feather(RUN / 'owned_temporal2fold_valid6270.feather')
    three = old.loc[old.month >= 65].copy()
    mask = old.month.to_numpy() >= 65
    three['tabm'] = np.mean(list(tabm.values()), axis=0)[mask]
    three['gru'] = np.mean([zunit(p[mask], old.month.to_numpy()[mask] <= 67) for p in gru.values()], axis=0)
    three.to_feather(RUN / 'owned_temporal3fold_valid6570.feather')
    (RUN / 'score_only.json').write_text(json.dumps({'status': 'complete', 'models_inferred': 12, 'temporal2fold_valid_months': '62-70 ex66', 'temporal3fold_valid_months': '65-70 ex66', 'tree_realmlp_transformer': 'existing strict train<=59 development predictions', 'no_training': True, 'public_weight': 0}, indent=2), encoding='utf-8')
    print('temporal validation recovery complete', flush=True)


if __name__ == '__main__':
    main()
