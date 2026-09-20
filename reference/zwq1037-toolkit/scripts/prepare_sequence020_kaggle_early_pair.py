"""Prepare a gated, matched early-window comparison on the existing notebook."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'


def main():
    source = (DIRECTORY / 'run_v21_subsecond_events_dev.py').read_text(encoding='utf-8')
    split = source.rindex("\nif __name__ == '__main__':")
    body, footer = source[:split], source[split:]
    tree = ast.parse(source)
    functions = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    run = functions['main'].replace('def main():', 'def _run_one():', 1)
    run = run.replace('months <= 59', 'months <= 47').replace('months >= 62', 'months >= 50').replace('months <= 70', 'months <= 59')
    run = run.replace("'train_end': 59", "'train_end': 47").replace("'valid_start': 62", "'valid_start': 50, 'valid_end': 59, 'model_kind': MODEL_KIND")
    run = run.replace("    for event_source in ('transaction', 'order'):\n        build_source('train', event_source, ids, CACHE)",
                      "    if MODEL_KIND == 'subsecond':\n        for event_source in ('transaction', 'order'):\n            build_source('train', event_source, ids, CACHE)")
    original_events = "events = {name: np.load(CACHE / f'{name}_events.npy', mmap_mode='r') for name in ('transaction', 'order')}"
    assert original_events in run
    run = run.replace(original_events,
        """events = ({name: np.load(CACHE / f'{name}_events.npy', mmap_mode='r') for name in ('transaction', 'order')}
              if MODEL_KIND == 'subsecond' else {
                  'transaction': np.memmap(GRID_INPUT / 'train_v2_tx_60x7.mmap', mode='r', dtype=np.float16, shape=(len(ids), 60, 7)),
                  'order': np.memmap(GRID_INPUT / 'train_v2_order_60x10.mmap', mode='r', dtype=np.float16, shape=(len(ids), 60, 10))})""")
    run = run.replace("lengths = {name: np.load(CACHE / f'{name}_lengths.npy') for name in events}",
                      "lengths = ({name: np.load(CACHE / f'{name}_lengths.npy') for name in events} if MODEL_KIND == 'subsecond' else {})")
    run = run.replace("    event_selected =", "    norm['market'] = _fit_grid_stats(market, np.sort(np.random.default_rng(SEED + 103).choice(train_rows, min(50000, len(train_rows)), replace=False)))\n    event_selected =")
    run = run.replace('norm[source] = fit_stats(events[source], lengths[source], event_selected)',
                      "norm[source] = (fit_stats(events[source], lengths[source], event_selected) if MODEL_KIND == 'subsecond' else _fit_grid_stats(events[source], event_selected))")
    run = run.replace("not SMOKE_BATCHES and best >= OLD_BASELINE + 0.001", "MODEL_KIND == 'subsecond' and not SMOKE_BATCHES and best >= OLD_BASELINE + 0.0003")
    run = run.replace("if SMOKE_BATCHES or (epoch == 1 and log['seconds'] * EPOCHS > 36000):", 'if SMOKE_BATCHES:')
    reference = ast.parse((DIRECTORY / 'run_v7_baseline.py').read_text(encoding='utf-8'))
    classes = {node.name: ast.unparse(node) for node in reference.body if isinstance(node, ast.ClassDef)}
    baseline = classes['_FactorizedStreamEncoder'] + '\n\n' + classes['_JointMultiStreamStaticModel']
    baseline = baseline.replace('_FactorizedStreamEncoder', '_EarlyGridEncoder').replace('_JointMultiStreamStaticModel', '_EarlyBaselineModel')
    baseline = baseline.replace('tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)',
        'safe_mask = padding_mask.clone()\n        safe_mask[padding_mask.all(dim=1), -1] = False\n        tokens = self.encoder(tokens, src_key_padding_mask=safe_mask.contiguous())')
    helpers = """
_EventSequenceDataset = SequenceDataset
_make_event_model = make_model

def _fit_grid_stats(values, rows):
    total = np.zeros(values.shape[-1], dtype=np.float64)
    squared = total.copy()
    count = total.copy()
    for start in range(0, len(rows), 256):
        block = np.asarray(values[rows[start:start + 256]], dtype=np.float32).reshape(-1, values.shape[-1])
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0)
        total += safe.sum(axis=0, dtype=np.float64)
        squared += np.square(safe).sum(axis=0, dtype=np.float64)
        count += finite.sum(axis=0)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(squared / np.maximum(count, 1) - mean * mean, 0))
    std[std < 1e-5] = 1
    return mean.astype(np.float32), std.astype(np.float32)

class _LegacySequenceDataset(_EventSequenceDataset):
    def __getitem__(self, position):
        row = self.rows[position]
        streams = []
        for name, values in (('market', self.market), ('transaction', self.events['transaction']), ('order', self.events['order'])):
            raw = np.asarray(values[row], dtype=np.float32)
            padding = np.abs(raw).sum(axis=-1) == 0
            block = np.clip(np.nan_to_num((raw - self.norm[name][0]) / self.norm[name][1]), -8, 8).astype(np.float32)
            block[padding] = 0
            streams.append(torch.from_numpy(block))
        static = np.asarray(self.static[row], dtype=np.float32)
        static = np.clip(np.nan_to_num((static - self.norm['static'][0]) / self.norm['static'][1]), -8, 8).astype(np.float32)
        return *streams, torch.from_numpy(static), torch.tensor(np.float32(self.target[row] / self.scale))

def make_model():
    return _make_event_model() if MODEL_KIND == 'subsecond' else _EarlyBaselineModel()

def main():
    global MODEL_KIND, RUN, SequenceDataset, OLD_BASELINE
    parent = Path('/kaggle/working/subsecond_early_pair')
    results = {}
    for kind in ('baseline', 'subsecond'):
        MODEL_KIND = kind
        RUN = parent / kind
        RUN.mkdir(parents=True, exist_ok=True)
        SequenceDataset = _EventSequenceDataset if kind == 'subsecond' else _LegacySequenceDataset
        OLD_BASELINE = results['baseline']['best_cosine'] if kind == 'subsecond' else 0.0
        _run_one()
        results[kind] = json.loads((RUN / 'result.json').read_text())
    delta = results['subsecond']['best_cosine'] - results['baseline']['best_cosine']
    summary = {'experiment': 'sequence020_early_matched_pair', 'train': '0-47', 'purged': [48,49],
               'validation': '50-59', 'seed': SEED, 'epochs_per_model': EPOCHS,
               'baseline': results['baseline']['best_cosine'], 'candidate': results['subsecond']['best_cosine'],
               'delta': delta, 'passed': delta >= 0.0003,
               'normalization_fit_on_train_only': True, 'training_precision': 'float32'}
    (parent / 'score_only.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
"""
    generated = '\n\n'.join((body, baseline, helpers, run, footer))
    compile(generated, 'run_v22_subsecond_early_pair.py', 'exec')
    script = DIRECTORY / 'run_v22_subsecond_early_pair.py'
    script.write_text(generated, encoding='utf-8')
    notebook = json.loads((DIRECTORY / 'run_v21_subsecond_events_dev.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source'] = generated.splitlines(keepends=True)
    path = DIRECTORY / 'run_v22_subsecond_early_pair.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'prepared': str(path), 'bytes': path.stat().st_size,
                      'launch_condition': 'local late-window delta >= 0.001',
                      'uploaded': False, 'models': ['old_factorized_transformer', 'subsecond_event_gru_transformer']}, indent=2))


if __name__ == '__main__':
    main()
