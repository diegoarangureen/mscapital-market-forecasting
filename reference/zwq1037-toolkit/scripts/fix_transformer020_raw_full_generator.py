"""Align full raw Transformer shuffling and resume state with EXP-020 dev."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
targets = [
    ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_fulltrain/run_v5_transformer_raw_full.py",
    ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v25_transformer020_raw_full.py",
]

for path in targets:
    text = path.read_text(encoding="utf-8")
    old_loader = '    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)'
    new_loader = '    generator = torch.Generator().manual_seed(SEED + 17)\n    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)'
    if old_loader in text:
        text = text.replace(old_loader, new_loader, 1)
    elif new_loader not in text:
        raise RuntimeError(f"Train loader marker missing: {path}")

    old_resume = '        start_epoch = int(checkpoint["next_epoch"])\n        print(f"resume_epoch={start_epoch}", flush=True)'
    new_resume = '        start_epoch = int(checkpoint["next_epoch"])\n        if "generator_state" in checkpoint:\n            generator.set_state(checkpoint["generator_state"].cpu())\n        print(f"resume_epoch={start_epoch}", flush=True)'
    if old_resume in text:
        text = text.replace(old_resume, new_resume, 1)
    elif new_resume not in text:
        raise RuntimeError(f"Resume marker missing: {path}")

    old_payload = '"next_epoch": epoch + 1, "logs": logs, "target_scale": target_scale}'
    new_payload = '"next_epoch": epoch + 1, "logs": logs, "target_scale": target_scale, "generator_state": generator.get_state()}'
    if old_payload in text:
        text = text.replace(old_payload, new_payload, 1)
    elif new_payload not in text:
        raise RuntimeError(f"Checkpoint payload marker missing: {path}")

    text = text.replace('checkpoint_path = WORK_DIR / "transformer_ema_training_checkpoint.pt"', 'checkpoint_path = WORK_DIR / "transformer_raw_training_checkpoint.pt"')
    text = text.replace('raise AssertionError("Invalid full-train EMA test predictions")', 'raise AssertionError("Invalid full-train raw test predictions")')
    compile(text, path.name, "exec")
    path.write_text(text, encoding="utf-8")

    notebook_path = path.with_suffix(".ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = text.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
