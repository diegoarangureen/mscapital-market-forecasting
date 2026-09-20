"""Edit the existing notebook for paired early-window sequence predictions."""
from pathlib import Path
import ast
import json

root = Path(__file__).resolve().parents[1]
directory = root / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'
base = (directory / 'run_v12_gru_temporal3fold.py').read_text(encoding='utf-8')
base_tree = ast.parse(base)
main_node = next(node for node in reversed(base_tree.body) if isinstance(node, ast.FunctionDef) and node.name == 'main')
source = '\n'.join(base.splitlines()[:main_node.lineno - 1]) + '\n'
source = source.replace('"/kaggle/working/factorized_gru_fulltrain"', '"/kaggle/working/owned_blend_early_confirmation"')
source = source.replace('os.environ.setdefault("OMP_NUM_THREADS", "2")', 'os.environ["OMP_NUM_THREADS"] = "2"\nos.environ["MKL_NUM_THREADS"] = "2"\nos.environ["OPENBLAS_NUM_THREADS"] = "2"')
old = ast.parse((directory / 'run_v7_baseline.py').read_text(encoding='utf-8'))
classes = {node.name: ast.unparse(node) for node in old.body if isinstance(node, ast.ClassDef)}
encoder = classes['_FactorizedStreamEncoder']
original = 'tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)'
assert encoder.count(original) == 1
encoder = encoder.replace(original, 'safe_mask = padding_mask.clone()\n        safe_mask[padding_mask.all(dim=1), -1] = False\n        tokens = self.encoder(tokens, src_key_padding_mask=safe_mask.contiguous())')
transformer = classes['_JointMultiStreamStaticModel'].replace('class _JointMultiStreamStaticModel(', 'class _OwnedFactorizedTransformer(')
source += '\n_OwnedTimeAwareGRU = _JointMultiStreamStaticModel\n' + encoder + '\n\n' + transformer + '\n'
source += '''
def main():
    global _STATIC_NORM
    require_runtime()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    train_ids, months, target = read_train_label()
    arrays = load_grid('train', train_ids.size)
    valid_idx = np.flatnonzero((months >= 50) & (months <= 59))
    device = torch.device('cuda')
    models = [('gru40', 40, _OwnedTimeAwareGRU), ('gru45', 45, _OwnedTimeAwareGRU),
              ('transformer47', 47, _OwnedFactorizedTransformer)]
    logs, predictions = {}, {}
    for name, cutoff, factory in models:
        seed_everything(SEED)
        train_idx = np.flatnonzero(months <= cutoff)
        norm = compute_norm(arrays, train_idx)
        _STATIC_NORM = _compute_static_norm(_STATIC_FEATURES, train_idx)
        target_scale = float(target[train_idx].std())
        train_ds = GridDataset(arrays, train_idx, norm, target=target, target_scale=target_scale)
        valid_ds = GridDataset(arrays, valid_idx, norm)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        core = factory().to(device)
        model = nn.DataParallel(core, device_ids=list(range(_VISIBLE_GPU_COUNT)))
        smoke = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            p = model(*[v.to(device).contiguous() for v in smoke[:4]])
        if not torch.isfinite(p).all():
            raise RuntimeError(f'{name} dual-GPU evaluation smoke failed')
        print(f'{name} dual_gpu_eval_smoke=passed train_end={cutoff} valid=50-59', flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        history = []
        for epoch in range(1, 5):
            started = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
            scheduler.step()
            row = {'epoch': epoch, 'loss': float(loss), 'seconds': time.time()-started}
            history.append(row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'cutoff': cutoff,
                        'target_scale': target_scale, 'stream_norm': norm, 'static_norm': _STATIC_NORM},
                       WORK_DIR / f'{name}_epoch_checkpoint.pt')
            print(name + ' ' + json.dumps(row), flush=True)
            status = {'status': 'running', 'current_model': name, 'epoch': epoch,
                      'completed_models': list(predictions), 'visible_gpu_count': _VISIBLE_GPU_COUNT,
                      'gpu_names': _VISIBLE_GPU_NAMES, 'public_weight': 0}
            (WORK_DIR / 'score_only.json').write_text(json.dumps(status, indent=2))
        prediction = predict(model, valid_loader, device) * target_scale
        if not np.isfinite(prediction).all():
            raise RuntimeError(f'{name} nonfinite validation predictions')
        predictions[name] = prediction
        logs[name] = history
        pd.DataFrame({'sample_id': train_ids[valid_idx], 'month': months[valid_idx],
                      'target': target[valid_idx], 'prediction': prediction}).to_csv(
                          WORK_DIR / f'{name}_validation_predictions.csv', index=False)
        del model, core, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
        gc.collect(); torch.cuda.empty_cache()
    result = {'status': 'complete', 'validation_months': '50-59',
              'train_ends': {'gru40': 40, 'gru45': 45, 'transformer47': 47},
              'normalization': 'Every stream and static normalizer fitted only on its model training rows',
              'visible_gpu_count': _VISIBLE_GPU_COUNT, 'gpu_names': _VISIBLE_GPU_NAMES,
              'epochs_per_model': 4, 'scores': {name: _cosine_score(target[valid_idx], p)
                                             for name, p in predictions.items()},
              'logs': logs, 'public_weight': 0, 'submission_status': 'not_submitted'}
    (WORK_DIR / 'score_only.json').write_text(json.dumps(result, indent=2))
    (WORK_DIR / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
'''
ast.parse(source)
archive = directory / 'run_v20_owned_blend_early_confirmation.py'
archive.write_text(source, encoding='utf-8')
(directory / 'run.py').write_text(source, encoding='utf-8')
notebook = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.splitlines(keepends=True)}],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 5}
(directory / 'run.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
print(archive)
