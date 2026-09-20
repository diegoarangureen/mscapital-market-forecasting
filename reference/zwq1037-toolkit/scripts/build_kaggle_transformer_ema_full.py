from pathlib import Path
import json
root = Path(r"F:\深度学习\projects\mscapital_market_forecasting")
base = root / r"data\interim\kaggle_kernels\multistream_factorized_transformer_fulltrain"
src = base / "run_v7_factorized_transformer_fulltrain.py"
out = base / "run_v4_transformer_ema_full.py"
text = src.read_text(encoding="utf-8")
text = text.replace('/kaggle/working/factorized_transformer', '/kaggle/working/transformer_ema_full')
text = text.replace('EPOCHS = int(os.environ.get("EPOCHS", "4"))', 'EPOCHS = int(os.environ.get("EPOCHS", "5"))', 1)
cut = text.rfind('\ndef main():')
if cut < 0:
    raise RuntimeError('main marker missing')
new_tail = r'''
EMA_DECAY = 0.999


def train_one_epoch_ema(model, loader, optimizer, scaler, device, ema_parameters):
    model.train()
    total = 0.0
    row_count = 0
    for batch in loader:
        market, transaction, order, static, target = batch
        market = market.to(device, non_blocking=True).contiguous()
        transaction = transaction.to(device, non_blocking=True).contiguous()
        order = order.to(device, non_blocking=True).contiguous()
        static = static.to(device, non_blocking=True).contiguous()
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            prediction = torch.nan_to_num(model(market, transaction, order, static), nan=0.0, posinf=0.0, neginf=0.0)
            loss = 0.35 * F.smooth_l1_loss(prediction, target) + 0.65 * cosine_loss(prediction, target)
        if not torch.isfinite(loss):
            print("skip non-finite loss batch", flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                ema_parameters[name].lerp_(parameter.detach(), 1.0 - EMA_DECAY)
        count = target.numel()
        total += float(loss.detach().cpu()) * count
        row_count += count
    return total / max(row_count, 1)


def main():
    global _STATIC_FEATURES, _STATIC_NORM
    require_runtime()
    seed_everything(SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if _VISIBLE_GPU_COUNT < 2:
        raise RuntimeError(f"Expected Kaggle T4 x2, found {_VISIBLE_GPU_COUNT} GPU(s)")
    print(f"transformer_ema_full epochs={EPOCHS}, ema={EMA_DECAY}, batch_size={BATCH_SIZE}, visible_gpu_count={_VISIBLE_GPU_COUNT}, gpu_names={_VISIBLE_GPU_NAMES}", flush=True)
    train_ids, months, target = read_train_label()
    train_arrays = load_grid("train", train_ids.size)
    train_idx = np.arange(train_ids.size, dtype=np.int64)
    norm = {name: {"mean": np.asarray(values["mean"], dtype=np.float32), "std": np.asarray(values["std"], dtype=np.float32)} for name, values in BUNDLED_STREAM_NORM.items()}
    save_norm(norm)
    target_scale = float(np.std(target))
    _STATIC_NORM = _compute_static_norm(_STATIC_FEATURES, train_idx)
    train_ds = GridDataset(train_arrays, train_idx, norm, target=target, target_scale=target_scale)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    device = torch.device("cuda")
    model = TransformerCnnModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    ema_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    logs = []
    start_epoch = 1
    checkpoint_path = WORK_DIR / "transformer_ema_training_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema_parameters = {name: value.to(device) for name, value in checkpoint["ema_parameters"].items()}
        logs = checkpoint["logs"]
        start_epoch = int(checkpoint["next_epoch"])
        print(f"resume_epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.time()
        loss = train_one_epoch_ema(model, train_loader, optimizer, scaler, device, ema_parameters)
        scheduler.step()
        row = {"epoch": epoch, "loss": float(loss), "seconds": time.time() - started}
        logs.append(row)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "ema_parameters": {name: value.detach().cpu() for name, value in ema_parameters.items()}, "next_epoch": epoch + 1, "logs": logs, "target_scale": target_scale}
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        print(f"epoch={epoch}/{EPOCHS} loss={loss:.6f} time={row['seconds']:.1f}s", flush=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ema_parameters[name])
    model_path = WORK_DIR / "factorized_transformer379_ema0999_full_e5.pt"
    torch.save({"model": model.state_dict(), "target_scale": target_scale, "epochs": EPOCHS, "seed": SEED, "ema_decay": EMA_DECAY}, model_path)
    del train_loader, train_ds, train_arrays
    gc.collect(); torch.cuda.empty_cache()
    test_ids = read_submission_ids()
    test_arrays = load_grid("test", test_ids.size)
    _STATIC_FEATURES = _load_test_static(test_ids)
    test_idx = np.arange(test_ids.size, dtype=np.int64)
    test_ds = GridDataset(test_arrays, test_idx, norm, target=None, target_scale=target_scale)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    prediction = predict(model, test_loader, device) * target_scale
    if prediction.shape != (test_ids.size,) or not np.isfinite(prediction).all():
        raise AssertionError("Invalid full-train EMA test predictions")
    submission_path = WORK_DIR / "standalone_factorized_transformer379_ema0999_full_e5.csv"
    pd.DataFrame({"sample_id": test_ids, "prediction": prediction}).to_csv(submission_path, index=False)
    result = {"experiment": "factorized_transformer379_ema0999_full", "status": "complete", "train_months": "all_available_0_70", "train_rows": int(train_ids.size), "test_rows": int(test_ids.size), "epochs": EPOCHS, "seed": SEED, "ema_decay": EMA_DECAY, "visible_gpu_count": int(_VISIBLE_GPU_COUNT), "gpu_names": _VISIBLE_GPU_NAMES, "loss": "0.35 SmoothL1 + 0.65 centered cosine", "prediction_mean": float(prediction.mean()), "prediction_std": float(prediction.std()), "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()), "prediction_file": submission_path.name, "checkpoint_file": model_path.name, "training_log": logs, "competition_submission_status": "prepared_not_submitted"}
    (WORK_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (WORK_DIR / "score_only.json").write_text(json.dumps({"status": "complete", "model": "factorized_transformer379", "ema_decay": EMA_DECAY, "offline_all_ex66": 0.1538447396, "offline_delta": -0.0072022482, "prediction_file": submission_path.name, "visible_gpu_count": int(_VISIBLE_GPU_COUNT)}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''
text = text[:cut] + new_tail
out.write_text(text, encoding="utf-8")
nb_path = out.with_suffix('.ipynb')
notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
nb_path.write_text(json.dumps(notebook), encoding="utf-8")
print(out)
print(nb_path)
