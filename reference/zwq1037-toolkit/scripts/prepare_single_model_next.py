"""Prepare isolated RealMLP385 and scheduled-loss Transformer experiments."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
KERNEL = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise AssertionError(f"Expected one anchor, got {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def prepare_realmlp_runner() -> None:
    # 隔离训练循环，保持历史基线文件不变。
    # Isolate the training loop while preserving the historical baseline.
    source = (SCRIPTS / "exp_realmlp_005_yunsu_public_reference.py").read_text(
        encoding="utf-8-sig"
    )
    source = replace_once(
        source,
        "    set_seed(SEED)\n    train_indices =",
        "    set_seed(SEED)\n"
        "    fold_dir = RUN_DIR / fold_name\n"
        "    fold_dir.mkdir(parents=True, exist_ok=True)\n"
        "    checkpoint_path = fold_dir / 'recovery_checkpoint.pt'\n"
        "    train_indices =",
    )
    source = replace_once(
        source,
        "    for epoch in range(1, epochs + 1):\n        model.train()",
        "    start_epoch = 1\n"
        "    if checkpoint_path.exists():\n"
        "        state = torch.load(checkpoint_path, map_location=device, weights_only=False)\n"
        "        if state['epochs'] != epochs or state['feature_names'] != kept_names:\n"
        "            raise RuntimeError('Resume configuration mismatch')\n"
        "        model.load_state_dict(state['model'])\n"
        "        optimizer.load_state_dict(state['optimizer'])\n"
        "        ema.state = state['ema']\n"
        "        generator.set_state(state['generator'].cpu())\n"
        "        torch.set_rng_state(state['torch_rng'].cpu())\n"
        "        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])\n"
        "        np.random.set_state(state['numpy_rng'])\n"
        "        random.setstate(state['python_rng'])\n"
        "        best_score, best_epoch = state['best_score'], state['best_epoch']\n"
        "        best_prediction = np.load(fold_dir / 'validation_predictions.npy')\n"
        "        history = state['history']\n"
        "        start_epoch = state['epoch'] + 1\n"
        "        print(f'RESUME next_epoch={start_epoch}', flush=True)\n"
        "    for epoch in range(start_epoch, epochs + 1):\n        model.train()",
    )
    source = replace_once(
        source,
        "        ema.restore(original)\n        row = {",
        "        if validation_score > best_score:\n"
        "            torch.save({'model': model.state_dict(), 'epoch': epoch,\n"
        "                        'score': validation_score, 'feature_names': kept_names},\n"
        "                       fold_dir / 'best_model.pt')\n"
        "        ema.restore(original)\n        row = {",
    )
    source = replace_once(
        source,
        "        print(\n            f\"EPOCH {fold_name}",
        "        np.save(fold_dir / 'validation_predictions.npy', best_prediction)\n"
        "        recovery = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),\n"
        "                    'ema': ema.state, 'generator': generator.get_state(),\n"
        "                    'torch_rng': torch.get_rng_state(),\n"
        "                    'cuda_rng': torch.cuda.get_rng_state_all(),\n"
        "                    'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),\n"
        "                    'epoch': epoch, 'epochs': epochs, 'feature_names': kept_names,\n"
        "                    'best_score': best_score, 'best_epoch': best_epoch, 'history': history}\n"
        "        torch.save(recovery, checkpoint_path.with_suffix('.tmp'))\n"
        "        checkpoint_path.with_suffix('.tmp').replace(checkpoint_path)\n"
        "        (fold_dir / 'score_only.json').write_text(json.dumps(\n"
        "            {'state': 'running', 'epoch': epoch, 'best_epoch': best_epoch,\n"
        "             'best_cosine': best_score}, indent=2), encoding='utf-8')\n"
        "        print(\n            f\"EPOCH {fold_name}",
    )
    compile(source, "realmlp011_reference.py", "exec")
    (SCRIPTS / "realmlp011_reference.py").write_text(source, encoding="utf-8")


def prepare_transformer() -> None:
    source = (KERNEL / "run_v7_baseline.py").read_text(encoding="utf-8-sig")
    source = replace_once(
        source,
        'WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working/factorized_transformer"))',
        'WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working/factorized_transformer_loss_schedule"))',
    )
    source = replace_once(
        source,
        '        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)',
        '        global CURRENT_COSINE_WEIGHT\n'
        '        CURRENT_COSINE_WEIGHT = cosine_weight_for_epoch(epoch)\n'
        '        print(f"loss_weights epoch={epoch} cosine={CURRENT_COSINE_WEIGHT:.3f} "\n'
        '              f"smooth_l1={1.0-CURRENT_COSINE_WEIGHT:.3f}", flush=True)\n'
        '        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)',
    )
    source = replace_once(
        source,
        '                0.35 * F.smooth_l1_loss(\n                    prediction, target\n                )\n'
        '                + 0.65 * cosine_loss(\n                    prediction, target\n                )',
        '                (1.0 - CURRENT_COSINE_WEIGHT) * F.smooth_l1_loss(\n'
        '                    prediction, target\n                )\n'
        '                + CURRENT_COSINE_WEIGHT * cosine_loss(\n'
        '                    prediction, target\n                )',
    )
    source = replace_once(
        source,
        '        "experiment": "multistream_factorized_transformer_dev",',
        '        "experiment": "factorized_transformer_late_loss_schedule",',
    )
    source = replace_once(
        source,
        '        "loss": "0.35 SmoothL1 + 0.65 centered cosine",',
        '        "loss": "cosine weights [0.65,0.65,0.65,0.50,0.35,0.20], SmoothL1=1-cosine",',
    )
    source = replace_once(
        source,
        '    best_cos = -1e9\n    best_epoch = 0',
        '    history = []\n    best_cos = -1e9\n    best_epoch = 0',
    )
    source = replace_once(
        source,
        '        score = cosine_np(pred_valid, target[valid_idx])',
        '        score = cosine_np(pred_valid, target[valid_idx])\n'
        '        history.append({"epoch": epoch, "loss": float(loss), "cosine": float(score),\n'
        '                        "cosine_weight": CURRENT_COSINE_WEIGHT, "seconds": time.time()-t0})',
    )
    source = replace_once(
        source,
        '    score_only = {\n        "best_cosine": float(final_score),\n        "best_epoch": best_epoch,\n    }',
        '    score_only = {\n        "best_cosine": float(final_score),\n'
        '        "best_epoch": best_epoch,\n'
        '        "baseline": 0.15538451490216448,\n'
        '        "delta": float(final_score)-0.15538451490216448,\n'
        '        "history": history,\n'
        '        "dual_gpu_smoke": dual_gpu_smoke_report,\n    }',
    )
    source = replace_once(
        source,
        '    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)',
        '    dual_gpu_smoke_report = verify_dual_gpu_participation(model, train_ds, device)\n'
        '    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)',
    )
    tree = ast.parse(source)
    nodes = [node for node in tree.body if not (
        isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
    )]
    body = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    extras = '''
CURRENT_COSINE_WEIGHT = 0.65

def cosine_weight_for_epoch(epoch):
    return [0.65, 0.65, 0.65, 0.50, 0.35, 0.20][epoch - 1]

def verify_dual_gpu_participation(model, dataset, device):
    # 用独立副本做一次双卡反传，保持正式模型和随机状态不变。
    # Check backward participation on a copy without changing training RNG/model.
    import copy
    state_cpu = torch.get_rng_state()
    state_cuda = torch.cuda.get_rng_state_all()
    core = copy.deepcopy(model.parallel.module)
    core.register_forward_hook(lambda module, args, output:
        setattr(module, '_smoke_device', str(output.device)))
    probe = nn.DataParallel(core, device_ids=[0, 1]).to(device)
    before = [torch.cuda.memory_allocated(i) for i in range(2)]
    batch = torch.utils.data.default_collate([dataset[i] for i in range(8)])
    inputs = [tensor.to(device) for tensor in batch[:4]]
    probe.train()
    output = probe(*inputs)
    output.square().mean().backward()
    for index in range(2):
        torch.cuda.synchronize(index)
    peaks = [torch.cuda.max_memory_allocated(i) for i in range(2)]
    report = {'devices': [0, 1], 'batch': 8,
              'allocated_before': before, 'peak_allocated': peaks,
              'both_devices_used': all(peak > initial for peak, initial in zip(peaks, before))}
    if not report['both_devices_used']:
        raise RuntimeError('Dual-GPU smoke failed')
    del probe, core, output, inputs, batch
    torch.cuda.empty_cache()
    torch.set_rng_state(state_cpu)
    torch.cuda.set_rng_state_all(state_cuda)
    (WORK_DIR / 'dual_gpu_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report

if __name__ == '__main__':
    torch.set_num_threads(2)
    try:
        main()
    except Exception as exc:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        (WORK_DIR / 'failure.json').write_text(json.dumps(
            {'exception': type(exc).__name__, 'message': str(exc)}, indent=2))
        raise
'''
    generated = body + "\n\n" + extras
    compile(generated, "run_v24_loss_schedule.py", "exec")
    script = KERNEL / "run_v24_loss_schedule.py"
    script.write_text(generated, encoding="utf-8")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": generated.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (KERNEL / "run_v24_loss_schedule.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metadata_path = KERNEL / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata["code_file"] = "run_v24_loss_schedule.ipynb"
    metadata["dataset_sources"] = [name for name in metadata["dataset_sources"]
                                   if name != "zwq1037/mscapital-strict-transformer-dev-v7"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    prepare_realmlp_runner()
    prepare_transformer()
    print("Prepared isolated RealMLP runner and Transformer loss-schedule notebook.")
