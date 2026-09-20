"""Prepare a resumable trainer without loading NumPy's competing BLAS runtime."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/exp_sequence_020_subsecond_gru_transformer.py').read_text(encoding='utf-8')
old_score = '''    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-20))'''
new_score = '''    x = torch.as_tensor(np.asarray(x, dtype=np.float64))
    y = torch.as_tensor(np.asarray(y, dtype=np.float64))
    return float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-20))'''
assert source.count(old_score) == 1
source = source.replace(old_score, new_score)
source = source.replace("OLD_BASELINE = 0.1553845145", "OLD_BASELINE = 0.1553845145\nSMOKE_BATCHES = int(os.environ.get('SEQ020_SMOKE_BATCHES', '0'))\nif SMOKE_BATCHES:\n    RUN = RUN.parent / (RUN.name + '-SMOKE')")
source = source.replace("    grid = ROOT /", "    if SMOKE_BATCHES:\n        valid_rows = valid_rows[:BATCH_SIZE]\n    grid = ROOT /", 1)
start = source.index('    best, logs = -1.0, []')
end = source.index("    (RUN / 'result.json').write_text", start)
replacement = '''    best, logs = -1.0, []
    recovery_path = RUN / 'recovery_checkpoint.pt'
    config = {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
              'smoke_batches': SMOKE_BATCHES, 'train_end': 59, 'valid_start': 62,
              'excluded_month': 66, 'cache': str(CACHE)}
    recovery = None
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location='cpu', weights_only=False)
        if recovery['config'] != config:
            raise RuntimeError('Checkpoint configuration mismatch')
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
                                      pin_memory=True, drop_last=True, generator=loader_generator)
            model.train()
            if not state:
                save_recovery(epoch, 0, 'train', 0, 0, 0)
            for batch_index, batch in enumerate(epoch_loader, start=skip):
                inputs = [x.to(device, non_blocking=True).contiguous() for x in batch[:4]]
                y = batch[4].to(device)
                optimizer.zero_grad(set_to_none=True)
                p = model(*inputs)
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
'''
source = source[:start] + replacement + source[end:]
destination = root / 'scripts/exp_sequence_020_subsecond_gru_transformer_resumable.py'
destination.write_text(source, encoding='utf-8')
print(destination)
