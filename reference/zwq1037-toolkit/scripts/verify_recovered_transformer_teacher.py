"""Reproduce a small stratified sample from the recovered strict validation model."""
import ast
import json
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v7_baseline.py'
ARTIFACTS = ROOT / 'outputs/factorized_transformer_dev_v7_exact'


def main():
    torch.set_num_threads(1)
    tree = ast.parse(REFERENCE.read_text(encoding='utf-8'))
    classes = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    namespace = {'torch': torch, 'nn': nn, 'np': np, 'SEED': 2026,
                 'MARKET_FEATURES': list(range(11)), 'TX_FEATURES': list(range(7)),
                 'ORDER_FEATURES': list(range(10)), 'MARKET_LEN': 200, 'FLOW_LEN': 60,
                 'STATIC_FEATURE_COUNT': 379}
    for name in ('ConvBlock', '_FactorizedStreamEncoder', '_JointMultiStreamStaticModel'):
        exec(classes[name], namespace)
    exec(functions['_compute_static_norm'], namespace)
    model = namespace['_JointMultiStreamStaticModel']().eval()
    checkpoint = torch.load(ARTIFACTS / 'best_transformer_cnn.pt', map_location='cpu', weights_only=False)
    state = {}
    for name, value in checkpoint['model'].items():
        while name.startswith(('module.', 'parallel.')):
            name = name.split('.', 1)[1]
        state[name] = value
    model.load_state_dict(state, strict=True)
    torch.backends.mha.set_fastpath_enabled(False)
    labels = pd.read_feather(ROOT / 'data/raw/label.feather', columns=['sample_id', 'month'])
    original = pd.read_csv(ROOT / 'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv', usecols=['sample_id', 'prediction'])
    chosen = []
    for month in (62, 63, 64, 65, 67, 68, 69, 70):
        month_rows = np.flatnonzero(labels.month.to_numpy() == month)
        chosen.extend(month_rows[[len(month_rows)//3, 2*len(month_rows)//3]].tolist())
    rows = np.asarray(chosen)
    static = np.load(ROOT / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')
    static_norm = namespace['_compute_static_norm'](static, np.flatnonzero(labels.month.to_numpy() <= 59))
    grid = ROOT / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid'
    norms = json.loads((ARTIFACTS / 'norm_v2.json').read_text())
    inputs = []
    for name, length, features in (('market', 200, 11), ('tx', 60, 7), ('order', 60, 10)):
        values = np.memmap(grid / f'train_v2_{name}_{length}x{features}.mmap', mode='r', dtype=np.float16, shape=(len(labels), length, features))
        raw = np.asarray(values[rows], dtype=np.float32)
        padding = np.abs(raw).sum(axis=-1) == 0
        mean, std = np.asarray(norms[name]['mean'], dtype=np.float32), np.asarray(norms[name]['std'], dtype=np.float32)
        block = np.clip(np.nan_to_num((raw-mean)/std), -8, 8).astype(np.float32)
        block[padding] = 0
        inputs.append(torch.from_numpy(block))
    normalized_static = np.clip(np.nan_to_num((np.asarray(static[rows], dtype=np.float32)-static_norm['mean'])/static_norm['std']), -8, 8).astype(np.float32)
    inputs.append(torch.from_numpy(normalized_static))
    with torch.no_grad():
        prediction = model(*inputs).numpy() * checkpoint['target_scale']
    expected = pd.DataFrame({'sample_id': labels.sample_id.to_numpy()[rows]}).merge(original, on='sample_id', validate='one_to_one').prediction.to_numpy()
    difference = prediction - expected
    relative_rmse = float(np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(expected**2)))
    report = {'rows': len(rows), 'months': [62,63,64,65,67,68,69,70],
              'source_score': float(checkpoint['score']), 'source_epoch': int(checkpoint['epoch']),
              'relative_prediction_rmse': relative_rmse, 'max_abs_difference': float(np.max(np.abs(difference))),
              'strict_state_loading': True, 'cpu_threads': 1, 'passed': relative_rmse < 0.005}
    (ROOT / 'outputs/submission_metadata/recovered_transformer_teacher_verification_20260917.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert report['passed'], 'Recovered teacher predictions differ from saved strict validation'
    np.savez(ARTIFACTS / 'reconstructed_static_normalization.npz', **static_norm)


if __name__ == '__main__':
    main()
