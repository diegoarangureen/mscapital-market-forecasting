"""Prepare a self-contained notebook using existing Kaggle input attachments."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'


def definitions(path, names):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = {node.name: ast.unparse(node) for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    return '\n\n'.join(found[name] for name in names)


def main():
    baseline = (DIRECTORY / 'run_v7_baseline.py').read_text(encoding='utf-8')
    base_tree = ast.parse(baseline)
    base_nodes = [node for node in base_tree.body if not isinstance(node, ast.If)
                  or ast.unparse(node.test) != "__name__ == '__main__'"]
    base = ast.unparse(ast.Module(body=base_nodes, type_ignores=[]))
    core = ROOT / 'scripts/subsecond_event_transformer_model.py'
    encoder = definitions(core, ['SubsecondEventEncoder'])
    classes = definitions(DIRECTORY / 'run_v7_baseline.py',
                          ['_FactorizedStreamEncoder', '_JointMultiStreamStaticModel'])
    classes = classes.replace(
        'tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)',
        'safe_mask = padding_mask.clone()\n        safe_mask[padding_mask.all(dim=1), -1] = False\n        tokens = self.encoder(tokens, src_key_padding_mask=safe_mask.contiguous())')
    classes = classes.replace(
        '_FactorizedStreamEncoder(len(TX_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout)',
        'SubsecondEventEncoder(9, d_model, dropout)').replace(
        '_FactorizedStreamEncoder(len(ORDER_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout)',
        'SubsecondEventEncoder(15, d_model, dropout)')
    assert classes.count('SubsecondEventEncoder(') == 2
    builder = definitions(ROOT / 'scripts/build_subsecond_event_cache.py',
                          ['encode_quantities', 'write_complete_samples', 'build_source'])
    builder = builder.replace("ROOT / f'data/raw/{split}/{source}.feather'",
                              "DATA / split / f'{source}.feather'")
    trainer_path = ROOT / 'scripts/exp_sequence_020_subsecond_gru_transformer_fp32.py'
    trainer = definitions(trainer_path, ['cosine', 'fit_stats', 'SequenceDataset',
                                         'prefetched_batches', 'predict', 'main'])
    trainer = trainer.replace("labels = pd.read_feather(ROOT / 'data/raw/label.feather')",
                              "label_ids, label_months, label_targets = read_train_label()\n    labels = pd.DataFrame({'sample_id': label_ids, 'month': label_months, 'target': label_targets})")
    trainer = trainer.replace("grid = ROOT / 'data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid/train_v2_market_200x11.mmap'",
                              "grid = GRID_INPUT / 'train_v2_market_200x11.mmap'")
    trainer = trainer.replace("static = np.load(ROOT / 'data/interim/our379_reference_cache/features.npy', mmap_mode='r')",
                              'static = _STATIC_FEATURES')
    start = trainer.index("    reference = ROOT /")
    end = trainer.index("    rng =", start)
    trainer = trainer[:start] + '    bundled = BUNDLED_STREAM_NORM\n' + trainer[end:]
    trainer = trainer.replace("    assert np.array_equal(ids, np.load(CACHE / 'sample_ids.npy'))",
        """    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select T4 x2; exactly two CUDA devices required')
    if not (DATA / 'train/transaction.feather').exists() or not (DATA / 'train/order.feather').exists():
        raise FileNotFoundError('Competition event inputs missing')
    CACHE.mkdir(parents=True, exist_ok=True)
    for event_source in ('transaction', 'order'):
        build_source('train', event_source, ids, CACHE)
    np.save(CACHE / 'sample_ids.npy', ids)
    assert np.array_equal(ids, np.load(CACHE / 'sample_ids.npy'))""")
    trainer = trainer.replace('model = make_model().to(device)',
        "model = torch.nn.DataParallel(make_model().to(device), device_ids=[0, 1])")
    trainer = trainer.replace("    print('real_batch_eval_smoke=passed', flush=True)",
        """    rng_before = torch.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state_all()
    model.train()
    smoke_prediction = model(*[x.to(device).contiguous() for x in smoke[:4]])
    smoke_prediction.square().mean().backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Nonfinite dual-GPU smoke gradients')
    model.zero_grad(set_to_none=True)
    used_memory = [torch.cuda.max_memory_allocated(i) for i in range(2)]
    if min(used_memory) < 1_000_000:
        raise RuntimeError('Both GPUs must participate in training and evaluation')
    torch.set_rng_state(rng_before)
    torch.cuda.set_rng_state_all(cuda_rng_before)
    print(json.dumps({'dual_gpu_eval_and_gradient_smoke': 'passed', 'allocated_bytes': used_memory}), flush=True)""")
    trainer = trainer.replace("log['seconds'] * EPOCHS > 7200", "log['seconds'] * EPOCHS > 36000")
    assert 'ROOT /' not in trainer and 'ROOT /' not in builder
    header = """import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['POLARS_MAX_THREADS'] = '1'
"""
    extra = """from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.backends.mha.set_fastpath_enabled(False)
torch.set_num_threads(2)
SEED = 2026
BATCH_SIZE = 256
EPOCHS = 4
SMOKE_BATCHES = 0
OLD_BASELINE = 0.1553845145
RUN = Path('/kaggle/working/subsecond_event_experiment')
CACHE = Path('/kaggle/working/subsecond_event_cache')
RUN.mkdir(parents=True, exist_ok=True)
"""
    footer = """def make_model():
    return _JointMultiStreamStaticModel()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (RUN / 'failure.json').write_text(json.dumps({'exception': type(exc).__name__, 'message': str(exc)}), encoding='utf-8')
        raise
    finally:
        # 缓存只用于本轮训练，输出保留分数、权重和验证预测。
        # Remove only explicitly named temporary cache files after this run.
        for name in ('transaction_events.npy', 'order_events.npy', 'transaction_lengths.npy', 'order_lengths.npy', 'sample_ids.npy'):
            path = CACHE / name
            if path.parent.resolve() != CACHE.resolve():
                raise RuntimeError('Cache cleanup path escaped its directory')
            path.unlink(missing_ok=True)
"""
    script = '\n\n'.join([header, base, extra, encoder, classes, builder, trainer, footer])
    compile(script, 'run_v21_subsecond_events_dev.py', 'exec')
    version = DIRECTORY / 'run_v21_subsecond_events_dev.py'
    version.write_text(script, encoding='utf-8')
    notebook = {'cells': [{'cell_type': 'code', 'execution_count': None,
                           'metadata': {}, 'outputs': [], 'source': script.splitlines(keepends=True)}],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}},
                'nbformat': 4, 'nbformat_minor': 5}
    path = DIRECTORY / 'run_v21_subsecond_events_dev.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    report = {'version': 21, 'script': str(version), 'notebook': str(path),
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'bytes': path.stat().st_size, 'compiled': True,
              'dual_gpu_runtime_smoke_required': True, 'training_precision': 'float32',
              'train_months': '0-59', 'purged_months': [60, 61], 'validation_months': '62-70 excluding 66',
              'events_built_on_kaggle_from_competition_input': True,
              'local_raw_data_or_weights_uploaded': False, 'uploaded': False}
    (ROOT / 'outputs/submission_metadata/kaggle_subsecond_v21_prepared_20260917.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
