"""Confirm frozen owned-blend weights on months50-59 with train-only preparation."""
from __future__ import annotations
import ast
import gc
import json
import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/interim/tree_experiments/EXP-BLEND-OWNED-20260917/early_confirmation'
MARKET6 = 'mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 realized_volatility_60 mid_momentum_60'.split()


def update_score(stage, **kwargs):
    path = RUN / 'score_only.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    previous.update({'status': stage, 'public_weight': 0, 'fixed_weights': {'tabm': .33, 'tree': 0, 'gru': .20, 'realmlp': .28, 'transformer': .19}, **kwargs})
    path.write_text(json.dumps(previous, indent=2), encoding='utf-8')


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available()
    RUN.mkdir(parents=True, exist_ok=True)
    source_cache = ROOT / 'data/interim/kaggle_relative319_dev'
    ids = np.load(source_cache / 'sample_ids.npy', mmap_mode='r')
    months = np.load(source_cache / 'months.npy', mmap_mode='r')
    target = np.load(source_cache / 'targets.npy', mmap_mode='r')
    static = np.load(ROOT / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')
    names379 = json.loads((ROOT / 'data/interim/our379_reference_cache/feature_columns.json').read_text(encoding='utf-8'))
    path385 = RUN / 'our385_features.npy'
    ready = RUN / 'our385_ready.json'
    if not ready.exists():
        market = pd.read_feather(ROOT / 'data/processed/train_market_microstructure_features.feather', columns=['sample_id', *MARKET6]).sort_values('sample_id')
        assert np.array_equal(market.sample_id.to_numpy(), ids)
        extra = market[MARKET6].to_numpy(np.float32)
        features = np.lib.format.open_memmap(path385, mode='w+', dtype=np.float32, shape=(len(ids), 385))
        for start in range(0, len(ids), 8192):
            stop = min(start + 8192, len(ids))
            features[start:stop, :379] = static[start:stop]
            features[start:stop, 379:] = extra[start:stop]
        features.flush()
        ready.write_text(json.dumps({'rows': len(ids), 'columns': names379 + MARKET6}), encoding='utf-8')
        del market, extra, features
        gc.collect()
    features = np.load(path385, mmap_mode='r')
    valid_rows = np.flatnonzero((months >= 50) & (months <= 59))
    valid_ids = ids[valid_rows]
    valid_features = np.asarray(features[valid_rows], np.float32)
    import train_tabm385_k32_temporal3fold_3seed as tabm
    tabm.RUN_DIR = RUN / 'tabm_checkpoints'
    tabm.PREDICTION_DIR = RUN / 'tabm_predictions'
    tabm.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in (40, 45):
        for seed in (42, 137, 2026):
            update_score('running_tabm', fold_end=fold, seed=seed, tabm_models_finished=len(results))
            prediction, metadata = tabm.train_one_model(fold, seed, features, valid_features, months,
                                                      target, names379 + MARKET6, valid_ids, torch.device('cuda'))
            results.append(prediction)
            gc.collect()
    average = np.mean(results, axis=0)
    pd.DataFrame({'sample_id': valid_ids, 'month': months[valid_rows], 'target': target[valid_rows], 'tabm': average}).to_feather(RUN / 'tabm_early_predictions.feather')
    del features, valid_features, results
    gc.collect()
    torch.cuda.empty_cache()

    update_score('running_realmlp', tabm_models_finished=6)
    real_path = RUN / 'realmlp_train047_valid5059/validation_predictions.npy'
    if not real_path.exists():
        import exp_realmlp_009_our379_corr095 as corr
        reference = corr.reference
        reference.select_features = corr.select_features_corr095
        reference.RUN_DIR = RUN
        # 固定使用晚期已选定的8轮，不在早期窗口重新选择轮数。
        # Freeze eight epochs selected previously on the late development window.
        reference_tree = ast.parse(Path(reference.__file__).read_text(encoding='utf-8'))
        node = next(n for n in reference_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_fold')
        function = ast.unparse(node)
        assert function.count('if validation_score > best_score:') == 1
        function = function.replace('if validation_score > best_score:', 'if epoch == epochs:')
        function = function.replace('original = ema.apply()', "checkpoint_dir = RUN_DIR / fold_name\n        checkpoint_dir.mkdir(parents=True, exist_ok=True)\n        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_dir / 'epoch_checkpoint.pt')\n        original = ema.apply()")
        exec(function, reference.__dict__)
        reference.run_fold('realmlp_train047_valid5059', 47, 50, 59, static, target, names379, ids, months, 8, False)
    real_ids = np.load(real_path.parent / 'validation_sample_ids.npy')
    assert np.array_equal(real_ids, valid_ids)
    pd.DataFrame({'sample_id': valid_ids, 'realmlp': np.load(real_path)}).to_feather(RUN / 'realmlp_early_predictions.feather')
    update_score('local_complete', tabm_models_finished=6, realmlp_epochs=8,
                 validation_months='50-59', tabm_train_ends=[40, 45], realmlp_train_end=47,
                 next='Combine with Kaggle GRU40/45 and Transformer47; evaluate frozen weights without another search')
    print('owned early local predictions complete', flush=True)


if __name__ == '__main__':
    main()
